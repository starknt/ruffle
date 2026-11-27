use super::{SocketAction, SocketHandle};
use async_channel::{Receiver, Sender};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Independent per-player budgets for network payloads awaiting VM delivery
/// and flushed payloads awaiting the socket writer. Script-owned buffers use
/// their normal Flash lifetime. Small queued allocations also consume credit.
pub const SOCKET_QUEUE_BYTES: u32 = 8 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum SocketQueueError {
    #[error("socket connection closed")]
    Closed,
    #[error("socket queue byte budget exhausted")]
    Full,
}

fn charge(len: usize) -> Result<u32, SocketQueueError> {
    u32::try_from(len.max(64)).map_err(|_| SocketQueueError::Full)
}

#[derive(Debug)]
pub struct SocketWrite {
    data: Vec<u8>,
    _credit: OwnedSemaphorePermit,
}

impl AsRef<[u8]> for SocketWrite {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

pub struct SocketOutput {
    sender: Sender<SocketWrite>,
    budget: SocketWriteBudget,
    xml_backlog: std::sync::Weak<std::sync::Mutex<xml::Backlog>>,
}

mod xml;

#[derive(Clone)]
pub struct SocketWriteBudget {
    permits: Arc<Semaphore>,
    limit: u32,
}

impl SocketWriteBudget {
    pub fn new(limit: u32) -> Self {
        assert!(limit > 0);
        Self {
            permits: Arc::new(Semaphore::new(limit as usize)),
            limit,
        }
    }
}

impl SocketOutput {
    pub fn channel(budget: SocketWriteBudget) -> (Self, Receiver<SocketWrite>) {
        let (sender, receiver) = async_channel::unbounded();
        (
            Self {
                sender,
                budget,
                xml_backlog: Default::default(),
            },
            receiver,
        )
    }

    /// A synchronous Flash flush must leave the buffer available on failure.
    pub fn try_write(&self, buffer: &mut Vec<u8>) -> Result<(), SocketQueueError> {
        if self.sender.is_closed() {
            return Err(SocketQueueError::Closed);
        }
        if buffer.is_empty() {
            return Ok(());
        }
        // Flash can flush an already allocated buffer larger than the whole window.
        // Transfer its allocation exclusively, so other sockets cannot accumulate
        // further writes until the writer has finished with it.
        let size = buffer.capacity().max(64).min(self.budget.limit as usize) as u32;
        let credit = self
            .budget
            .permits
            .clone()
            .try_acquire_many_owned(size)
            .map_err(|_| SocketQueueError::Full)?;
        let write = SocketWrite {
            data: std::mem::take(buffer),
            _credit: credit,
        };
        self.sender.try_send(write).map_err(|error| {
            *buffer = error.into_inner().data;
            SocketQueueError::Closed
        })
    }

    pub fn close(&self) -> bool {
        self.sender.close()
    }
}

struct QueuedAction {
    action: SocketAction,
    _credit: Option<OwnedSemaphorePermit>,
}

#[derive(Clone)]
pub struct SocketActionSender {
    sender: Sender<QueuedAction>,
    budget: Arc<Semaphore>,
    limit: u32,
}

pub struct SocketActionReceiver {
    receiver: Receiver<QueuedAction>,
    budget: Arc<Semaphore>,
}

impl SocketActionSender {
    pub fn channel(limit: u32) -> (Self, SocketActionReceiver) {
        let (sender, receiver) = async_channel::unbounded();
        let budget = Arc::new(Semaphore::new(limit as usize));
        (
            Self {
                sender,
                budget: budget.clone(),
                limit,
            },
            SocketActionReceiver { receiver, budget },
        )
    }

    fn payload_charge(&self, action: &SocketAction) -> Result<Option<u32>, SocketQueueError> {
        let SocketAction::Data(_, bytes) = action else {
            return Ok(None);
        };
        let size = charge(bytes.capacity())?;
        if size > self.limit {
            return Err(SocketQueueError::Full);
        }
        Ok(Some(size))
    }

    fn enqueue(
        &self,
        action: SocketAction,
        credit: Option<OwnedSemaphorePermit>,
    ) -> Result<(), SocketQueueError> {
        self.sender
            .try_send(QueuedAction {
                action,
                _credit: credit,
            })
            .map_err(|_| SocketQueueError::Closed)
    }

    pub fn try_send(&self, action: SocketAction) -> Result<(), SocketQueueError> {
        let credit = self
            .payload_charge(&action)?
            .map(|size| self.budget.clone().try_acquire_many_owned(size))
            .transpose()
            .map_err(|_| SocketQueueError::Full)?;
        self.enqueue(action, credit)
    }

    pub async fn send(&self, action: SocketAction) -> Result<(), SocketQueueError> {
        let credit = match self.payload_charge(&action)? {
            Some(size) => Some(
                self.budget
                    .clone()
                    .acquire_many_owned(size)
                    .await
                    .map_err(|_| SocketQueueError::Closed)?,
            ),
            None => None,
        };
        self.enqueue(action, credit)
    }

    pub(super) fn notify_close(&self, handle: SocketHandle) {
        let _ = self.try_send(SocketAction::Close(handle));
    }
}

impl SocketActionReceiver {
    pub(super) fn ready_count(&self) -> usize {
        self.receiver.len()
    }
    pub fn try_recv(&self) -> Result<SocketAction, async_channel::TryRecvError> {
        self.receiver.try_recv().map(|queued| queued.action)
    }
    pub async fn recv(&self) -> Result<SocketAction, async_channel::RecvError> {
        self.receiver.recv().await.map(|queued| queued.action)
    }
}

impl Drop for SocketActionReceiver {
    fn drop(&mut self) {
        self.budget.close();
    }
}
