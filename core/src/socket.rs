use crate::avm1::{
    Activation as Avm1Activation, ActivationIdentifier, ExecutionReason, Object as Avm1Object,
    globals::xml_socket::XmlSocket,
};
use crate::avm2::object::{EventObject, SocketObject};
use crate::avm2::{Activation as Avm2Activation, Avm2};
use crate::backend::navigator::NavigatorBackend;
use crate::context::UpdateContext;
use crate::string::AvmString;

use gc_arena::Collect;
use ruffle_macros::istr;
use slotmap::{SlotMap, new_key_type};
use std::{cell::Cell, time::Duration};

mod transport;
pub use transport::{
    SOCKET_QUEUE_BYTES, SocketActionReceiver, SocketActionSender, SocketOutput, SocketQueueError,
    SocketWrite, SocketWriteBudget,
};

new_key_type! {
    pub struct SocketHandle;
}

#[derive(Copy, Clone, Collect)]
#[collect(no_drop)]
enum SocketKind<'gc> {
    Avm2(SocketObject<'gc>),
    Avm1(Avm1Object<'gc>),
}

#[derive(Collect)]
#[collect(no_drop)]
struct Socket<'gc> {
    target: SocketKind<'gc>,
    #[collect(require_static)]
    sender: SocketOutput,
    connected: Cell<bool>,
}

impl<'gc> Socket<'gc> {
    fn new(target: SocketKind<'gc>, sender: SocketOutput) -> Self {
        Self {
            target,
            sender,
            connected: Cell::new(false),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ConnectionState {
    Connected,
    Failed,
    TimedOut,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SocketAction {
    Connect(SocketHandle, ConnectionState),
    Data(SocketHandle, Vec<u8>),
    Close(SocketHandle),
}

/// Capture the number of actions that were ready when a frame started.
/// Actions produced by ActionScript callbacks are deferred to the next frame,
/// matching the previous drain-then-dispatch ordering without retaining a
/// second Vec of all pending payloads.
fn socket_action_budget(receiver: &SocketActionReceiver) -> usize {
    receiver.ready_count()
}

/// Manages the collection of Sockets.
#[derive(Collect)]
#[collect(no_drop)]
pub struct Sockets<'gc> {
    sockets: SlotMap<SocketHandle, Socket<'gc>>,

    #[collect(require_static)]
    receiver: SocketActionReceiver,
    #[collect(require_static)]
    sender: SocketActionSender,
    #[collect(require_static)]
    output_budget: SocketWriteBudget,
}

impl<'gc> Sockets<'gc> {
    pub fn empty() -> Self {
        let (sender, receiver) = SocketActionSender::channel(SOCKET_QUEUE_BYTES);

        Self {
            sockets: SlotMap::with_key(),
            receiver,
            sender,
            output_budget: SocketWriteBudget::new(SOCKET_QUEUE_BYTES),
        }
    }

    pub fn connect_avm2(
        &mut self,
        backend: &mut dyn NavigatorBackend,
        target: SocketObject<'gc>,
        host: String,
        port: u16,
    ) {
        let (sender, receiver) = SocketOutput::channel(self.output_budget.clone());

        let socket = Socket::new(SocketKind::Avm2(target), sender);
        let handle = self.sockets.insert(socket);

        // NOTE: This call will send SocketAction::Connect to sender with connection status.
        backend.connect_socket(
            sanitize_host(&host).to_string(),
            port,
            Duration::from_millis(target.timeout().into()),
            handle,
            receiver,
            self.sender.clone(),
        );

        if let Some(existing_handle) = target.set_handle(handle) {
            // As written in the AS3 docs, we are supposed to close the existing connection,
            // when a new one is created.
            self.close(existing_handle)
        }
    }

    pub fn connect_avm1(
        &mut self,
        backend: &mut dyn NavigatorBackend,
        target: Avm1Object<'gc>,
        host: String,
        port: u16,
    ) {
        let (sender, receiver) = SocketOutput::channel(self.output_budget.clone());

        let xml_socket = match XmlSocket::cast(target.into()) {
            Some(xml_socket) => xml_socket,
            None => return,
        };

        let socket = Socket::new(SocketKind::Avm1(target), sender);
        let handle = self.sockets.insert(socket);

        // NOTE: This call will send SocketAction::Connect to sender with connection status.
        backend.connect_socket(
            sanitize_host(&host).to_string(),
            port,
            Duration::from_millis(xml_socket.timeout().into()),
            handle,
            receiver,
            self.sender.clone(),
        );

        if let Some(existing_handle) = xml_socket.set_handle(handle) {
            // NOTE: AS2 docs don't specify what happens when connect is called with open connection,
            //       but we will close the existing connection anyway.
            self.close(existing_handle)
        }
    }

    pub fn is_connected(&self, handle: SocketHandle) -> bool {
        if let Some(socket) = self.sockets.get(handle) {
            socket.connected.get()
        } else {
            false
        }
    }

    pub fn send(&self, handle: SocketHandle, buffer: &mut Vec<u8>) -> Result<(), SocketQueueError> {
        self.sockets
            .get(handle)
            .ok_or(SocketQueueError::Closed)?
            .sender
            .try_write(buffer)
    }

    pub fn fail_xml_write(&self, handle: SocketHandle) {
        if let Some(socket) = self.sockets.get(handle)
            && socket.sender.close()
        {
            self.sender.notify_close(handle);
        }
    }

    pub fn close_all(&mut self) {
        for (_, socket) in self.sockets.drain() {
            Self::close_internal(socket);
        }
    }

    pub fn close(&mut self, handle: SocketHandle) {
        if let Some(socket) = self.sockets.remove(handle) {
            Self::close_internal(socket);
        }
    }

    fn close_internal(socket: Socket) {
        let Socket {
            sender,
            target,
            connected: _,
        } = socket;

        drop(sender); // NOTE: By dropping the sender, the reading task will close automatically.

        // Clear the buffers if the connection was closed.
        match target {
            SocketKind::Avm1(target) => {
                let target = XmlSocket::cast(target.into()).expect("target should be XmlSocket");

                target.read_buffer().clear();
            }
            SocketKind::Avm2(target) => {
                target.read_buffer().clear();
                target.write_buffer().clear();
            }
        }
    }

    pub fn update_sockets(context: &mut UpdateContext<'gc>) {
        let action_budget = {
            let _receive =
                ruffle_render::profile::time(ruffle_render::profile::Phase::SocketReceive);
            socket_action_budget(&context.sockets.receiver)
        };

        for _ in 0..action_budget {
            let Ok(action) = context.sockets.receiver.try_recv() else {
                break;
            };
            match &action {
                SocketAction::Connect(..) => ruffle_render::profile::socket_connect(),
                SocketAction::Data(_, data) => ruffle_render::profile::socket_data(data.len()),
                SocketAction::Close(..) => ruffle_render::profile::socket_close(),
            }
            match action {
                SocketAction::Connect(handle, ConnectionState::Connected) => {
                    let _connect =
                        ruffle_render::profile::time(ruffle_render::profile::Phase::SocketConnect);
                    let target = match context.sockets.sockets.get(handle) {
                        Some(socket) => {
                            socket.connected.set(true);
                            socket.target
                        }
                        // Socket must have been closed before we could send event.
                        None => continue,
                    };

                    match target {
                        SocketKind::Avm2(target) => {
                            let activation = Avm2Activation::from_nothing(context);

                            let connect_evt =
                                EventObject::bare_default_event(activation.context, "connect");
                            Avm2::dispatch_event(activation.context, connect_evt, target.into());
                        }
                        SocketKind::Avm1(target) => {
                            let mut activation = Avm1Activation::from_stub(
                                context,
                                ActivationIdentifier::root("[XMLSocket]"),
                            );

                            let _ = target.call_method(
                                istr!("onConnect"),
                                &[true.into()],
                                &mut activation,
                                ExecutionReason::Special,
                            );
                        }
                    }
                }
                SocketAction::Connect(
                    handle,
                    ConnectionState::Failed | ConnectionState::TimedOut,
                ) => {
                    let _connect =
                        ruffle_render::profile::time(ruffle_render::profile::Phase::SocketConnect);
                    let target = match context.sockets.sockets.get(handle) {
                        Some(socket) => socket.target,
                        // Socket must have been closed before we could send event.
                        None => continue,
                    };

                    match target {
                        SocketKind::Avm2(target) => {
                            let mut activation = Avm2Activation::from_nothing(context);

                            let io_error_evt = EventObject::io_error_event(
                                &mut activation,
                                "Error #2031: Socket Error.",
                                2031,
                            );

                            Avm2::dispatch_event(activation.context, io_error_evt, target.into());
                        }
                        // TODO: Not sure if avm1 xmlsocket has a way to notify a error. (Probably should just fire connect event with success as false).
                        SocketKind::Avm1(target) => {
                            let mut activation = Avm1Activation::from_stub(
                                context,
                                ActivationIdentifier::root("[XMLSocket]"),
                            );

                            let _ = target.call_method(
                                istr!("onConnect"),
                                &[false.into()],
                                &mut activation,
                                ExecutionReason::Special,
                            );
                        }
                    }
                }
                SocketAction::Data(handle, data) => {
                    let _data =
                        ruffle_render::profile::time(ruffle_render::profile::Phase::SocketData);
                    let target = match context.sockets.sockets.get(handle) {
                        Some(socket) => socket.target,
                        // Socket must have been closed before we could send event.
                        None => continue,
                    };

                    match target {
                        SocketKind::Avm2(target) => {
                            let mut activation = Avm2Activation::from_nothing(context);

                            let bytes_loaded = data.len();
                            target.read_buffer().extend(data);

                            let progress_evt = EventObject::progress_event(
                                &mut activation,
                                "socketData",
                                bytes_loaded,
                                0, // NOTE: bytesTotal is not used by socketData event.
                            );

                            Avm2::dispatch_event(activation.context, progress_evt, target.into());
                        }
                        SocketKind::Avm1(target) => {
                            let mut activation = Avm1Activation::from_stub(
                                context,
                                ActivationIdentifier::root("[XMLSocket]"),
                            );

                            // NOTE: This is enforced in connect_avm1() function.
                            let xml_socket =
                                XmlSocket::cast(target.into()).expect("target should be XmlSocket");

                            // The read buffer should never contain null bytes at this point,
                            // since they are always processed immediately. Therefore, we
                            // only need to check the new data for null bytes.
                            let has_null = data.contains(&0);
                            xml_socket.read_buffer().extend(data);

                            if has_null {
                                // Process complete messages (null-terminated) one at a time.
                                // We release the buffer borrow before each AS call to avoid
                                // conflicts if AS code accesses the socket.
                                loop {
                                    let message = {
                                        let buffer = &mut *xml_socket.read_buffer();
                                        match buffer.iter().position(|&b| b == 0) {
                                            Some(pos) => {
                                                let msg: Vec<u8> = buffer.drain(..=pos).collect();
                                                Some(AvmString::new_utf8_bytes(
                                                    activation.gc(),
                                                    &msg[..msg.len() - 1],
                                                ))
                                            }
                                            None => None,
                                        }
                                    };

                                    match message {
                                        Some(msg) => {
                                            let _ = target.call_method(
                                                istr!("onData"),
                                                &[msg.into()],
                                                &mut activation,
                                                ExecutionReason::Special,
                                            );
                                        }
                                        None => break,
                                    }
                                }
                            }
                        }
                    }
                }
                SocketAction::Close(handle) => {
                    let _close =
                        ruffle_render::profile::time(ruffle_render::profile::Phase::SocketClose);
                    let target = match context.sockets.sockets.remove(handle) {
                        Some(socket) => {
                            socket.connected.set(false);
                            socket.target
                        }
                        // Socket must have been closed before we could send event.
                        None => continue,
                    };

                    match target {
                        SocketKind::Avm2(target) => {
                            // Clear the buffers if the connection was closed.
                            target.read_buffer().clear();
                            target.write_buffer().clear();

                            let close_evt = EventObject::bare_default_event(context, "close");
                            Avm2::dispatch_event(context, close_evt, target.into());
                        }
                        SocketKind::Avm1(target) => {
                            let mut activation = Avm1Activation::from_stub(
                                context,
                                ActivationIdentifier::root("[XMLSocket]"),
                            );

                            // Clear the read buffer if the connection was closed.
                            let socket =
                                XmlSocket::cast(target.into()).expect("target should be XmlSocket");

                            socket.read_buffer().clear();

                            let _ = target.call_method(
                                istr!("onClose"),
                                &[],
                                &mut activation,
                                ExecutionReason::Special,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Flash treats a socket host as a cstring, and stops reading at a null byte.
/// We need to account for this here.
fn sanitize_host(host: &str) -> &str {
    host.split('\0').next().unwrap()
}

#[cfg(test)]
mod tests {
    use super::{ConnectionState, SocketAction, SocketHandle, sanitize_host, socket_action_budget};

    #[test]
    fn truncate_host_at_null() {
        assert_eq!(
            sanitize_host("1.2.3.4\0nonsense that gets dropped\0"),
            "1.2.3.4"
        );
        assert_eq!(sanitize_host("\0nonsense"), "");
        assert_eq!(sanitize_host("host\0"), "host");
    }

    #[test]
    fn normal_host() {
        assert_eq!(sanitize_host("1.2.3.4"), "1.2.3.4");
    }

    #[test]
    fn a_slow_socket_writer_retains_credits_until_the_write_finishes() {
        let budget = super::SocketWriteBudget::new(128);
        let (send, take) = super::SocketOutput::channel(budget.clone());
        let (other, _other_reader) = super::SocketOutput::channel(budget);
        let mut first = vec![1; 128];
        send.try_write(&mut first).unwrap();
        assert!(first.is_empty());
        let writing = take.try_recv().unwrap();
        let mut second = vec![2; 65];
        assert!(send.try_write(&mut second).is_err());
        assert_eq!(second, vec![2; 65]);
        assert!(other.try_write(&mut second).is_err());
        assert_eq!(writing.as_ref(), &[1; 128]);
        drop(writing);
        send.try_write(&mut second).unwrap();
        drop(send);
        assert_eq!(take.try_recv().unwrap().as_ref(), &[2; 65]);
        assert!(take.try_recv().is_err());
    }

    #[test]
    fn a_large_flash_flush_can_retry_and_owns_the_whole_writer_window() {
        let budget = super::SocketWriteBudget::new(128);
        let (a, reader) = super::SocketOutput::channel(budget.clone());
        let (b, other_reader) = super::SocketOutput::channel(budget);
        a.try_write(&mut vec![1]).unwrap();
        let mut large = vec![7; 513];
        assert!(b.try_write(&mut large).is_err());
        assert_eq!(large, [7; 513]);
        drop(reader.try_recv().unwrap());
        b.try_write(&mut large).unwrap();
        assert!(large.is_empty());
        let writing = other_reader.try_recv().unwrap();
        assert!(a.try_write(&mut vec![2]).is_err());
        assert_eq!(writing.as_ref(), [7; 513]);
        drop(writing);
        a.try_write(&mut vec![2]).unwrap();
        assert_eq!(reader.try_recv().unwrap().as_ref(), [2]);
    }

    #[test]
    fn paused_socket_input_resumes_on_delivery_and_wakes_on_session_close() {
        use futures::FutureExt;
        let handle = SocketHandle::default();
        let (events, inbox) = super::SocketActionSender::channel(128);
        events
            .try_send(SocketAction::Data(handle, vec![1; 128]))
            .unwrap();
        let mut reading = Box::pin(events.send(SocketAction::Data(handle, vec![2; 128])));
        assert!(reading.as_mut().now_or_never().is_none());
        events
            .try_send(SocketAction::Connect(handle, ConnectionState::Failed))
            .unwrap();
        assert_eq!(
            inbox.try_recv().unwrap(),
            SocketAction::Data(handle, vec![1; 128])
        );
        assert!(reading.as_mut().now_or_never().unwrap().is_ok());
        assert_eq!(
            inbox.try_recv().unwrap(),
            SocketAction::Connect(handle, ConnectionState::Failed)
        );
        let mut pending = Box::pin(events.send(SocketAction::Data(handle, vec![3; 128])));
        assert!(pending.as_mut().now_or_never().is_none());
        drop(inbox);
        assert!(pending.as_mut().now_or_never().unwrap().is_err());
    }

    #[test]
    fn socket_action_budget_is_a_frame_snapshot() {
        let (sender, receiver) = super::SocketActionSender::channel(128);
        sender
            .try_send(SocketAction::Connect(
                SocketHandle::default(),
                ConnectionState::Connected,
            ))
            .unwrap();
        let budget = socket_action_budget(&receiver);
        sender
            .try_send(SocketAction::Close(SocketHandle::default()))
            .unwrap();
        assert_eq!(budget, 1);
        assert_eq!(receiver.ready_count(), 2);
    }
}
