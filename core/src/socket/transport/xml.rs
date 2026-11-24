use super::{SocketOutput, SocketQueueError, SocketWrite, SocketWriteBudget};
use crate::backend::tasks::OwnedFuture;
use async_channel::Sender;
use futures::future::{Either, select};
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// File-backed FIFO for a congested XMLSocket. Script has no flush
/// result to retry. Only the bytes entering the transport occupy its RAM budget.
pub(super) struct Backlog {
    directory: Option<PathBuf>,
    pending: Option<File>,
    len: u64,
    finished: bool,
}

impl Backlog {
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        if self.pending.is_none() {
            self.pending = Some(match &self.directory {
                Some(path) => tempfile::tempfile_in(path)?,
                None => tempfile::tempfile()?,
            });
        }
        self.pending.as_mut().unwrap().write_all(bytes)?;
        self.len += bytes.len() as u64;
        Ok(())
    }
}

impl SocketOutput {
    pub(in crate::socket) fn write_xml(
        &mut self,
        bytes: &mut Vec<u8>,
        directory: Option<&Path>,
    ) -> io::Result<Option<OwnedFuture<(), io::Error>>> {
        if self.sender.is_closed() {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        if let Some(shared) = self.xml_backlog.upgrade() {
            let mut backlog = shared.lock().unwrap();
            if !backlog.finished {
                backlog.append(bytes)?;
                return Ok(None);
            }
        }
        match self.try_write(bytes) {
            Ok(()) => return Ok(None),
            Err(SocketQueueError::Closed) => return Err(io::ErrorKind::BrokenPipe.into()),
            Err(SocketQueueError::Full) => {}
        }
        let mut backlog = Backlog {
            directory: directory.map(Path::to_owned),
            pending: None,
            len: 0,
            finished: false,
        };
        backlog.append(bytes)?;
        let shared = Arc::new(Mutex::new(backlog));
        self.xml_backlog = Arc::downgrade(&shared);
        let budget = self.budget.clone();
        let writer = self.sender.clone();
        Ok(Some(Box::pin(async move {
            let result = drain(shared, budget, &writer).await;
            if result.is_err() {
                writer.close();
            }
            result
        })))
    }
}

async fn drain(
    shared: Arc<Mutex<Backlog>>,
    budget: SocketWriteBudget,
    writer: &Sender<SocketWrite>,
) -> io::Result<()> {
    loop {
        let (mut file, mut remaining) = {
            let mut backlog = shared.lock().unwrap();
            let Some(file) = backlog.pending.take() else {
                backlog.finished = true;
                return Ok(());
            };
            (file, std::mem::take(&mut backlog.len))
        };
        // New sends append to the next file while this one drains. Its disk
        // space is released even on a continuously busy connection.
        file.rewind()?;
        while remaining != 0 {
            let size = remaining.min(u64::from(budget.limit.min(64 * 1024))) as u32;
            let reserve = budget
                .permits
                .clone()
                .acquire_many_owned(size.max(64).min(budget.limit));
            futures::pin_mut!(reserve);
            let closed = writer.closed();
            futures::pin_mut!(closed);
            let credit = match select(reserve, closed).await {
                Either::Left((result, _)) => result.map_err(io::Error::other)?,
                Either::Right(_) => return Ok(()),
            };
            let mut data = vec![0; size as usize];
            file.read_exact(&mut data)?;
            if writer
                .try_send(SocketWrite {
                    data,
                    _credit: credit,
                })
                .is_err()
            {
                return Ok(());
            }
            remaining -= u64::from(size);
        }
    }
}
