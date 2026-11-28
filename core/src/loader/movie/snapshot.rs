//! A synchronous loadBytes call must preserve its argument even when its VM
//! cannot poll a loader. Spill only excess snapshots, independently of that VM.
use super::{MovieBudgets, MovieInput};
use bytes::Bytes;
use futures::channel::oneshot;
use std::fs::File;
use std::io::{self, Read, Seek, Write};
use std::sync::{Arc, Mutex, mpsc};

type WriteJob = (Bytes, mpsc::SyncSender<io::Result<File>>);

#[derive(Clone, Default)]
pub(super) struct SnapshotWriter {
    sender: Arc<Mutex<Option<mpsc::SyncSender<WriteJob>>>>,
    directory: Option<std::path::PathBuf>,
}

impl SnapshotWriter {
    fn store(&self, bytes: Bytes) -> io::Result<File> {
        let writer = {
            let mut current = self.sender.lock().unwrap();
            if current.is_none() {
                // Rendezvous bounds outstanding input to the writer's current
                // snapshot. Its progress never needs the player or a GPU callback.
                let directory = self.directory.clone();
                let (tx, rx) = mpsc::sync_channel::<WriteJob>(0);
                std::thread::Builder::new()
                    .name("ruffle-loadbytes".into())
                    .spawn(move || {
                        for (bytes, reply) in rx {
                            let result = (|| {
                                let mut file = match &directory {
                                    Some(directory) => tempfile::tempfile_in(directory),
                                    None => tempfile::tempfile(),
                                }?;
                                file.write_all(&bytes)?;
                                file.rewind()?;
                                Ok(file)
                            })();
                            drop(bytes);
                            let _ = reply.send(result);
                        }
                    })?;
                *current = Some(tx);
            }
            current.as_ref().unwrap().clone()
        };
        // The caller already owns this snapshot. Finish spooling before it
        // returns to AS and can mutate or enqueue another unadmitted body.
        let (reply, receiver) = mpsc::sync_channel(0);
        writer.send((bytes, reply)).map_err(io::Error::other)?;
        receiver.recv().map_err(io::Error::other)?
    }
}

impl MovieBudgets {
    pub(in crate::loader) fn set_snapshot_directory(&mut self, directory: std::path::PathBuf) {
        self.spill = SnapshotWriter {
            directory: Some(directory),
            ..Default::default()
        };
    }

    pub(in crate::loader) fn capture_snapshot(
        &self,
        bytes: Bytes,
    ) -> impl Future<Output = io::Result<MovieInput>> + use<> {
        let length = bytes.len();
        let input = self.input.clone();
        let reservation = input
            .permits
            .clone()
            .try_acquire_many_owned(input.admission_units(Some(length)));
        let snapshot = match reservation {
            Ok(reservation) => Ok(MovieInput {
                allows_movie_cache: false,
                bytes,
                loader_request: None,
                abort: None,
                reservation: super::ByteCharge {
                    admission: input.scope.admission(input.kind, length as u64),
                    permits: reservation,
                    bytes: length,
                },
            }),
            Err(_) => Err(self.spill.store(bytes)),
        };
        async move {
            let mut file = match snapshot {
                Ok(input) => return Ok(input),
                Err(stored) => stored?,
            };
            let reservation = input.reserve(Some(length)).await;
            let (send, receive) = oneshot::channel();
            rayon::spawn_fifo(move || {
                if !send.is_canceled() {
                    let result = (|| {
                        let mut data = vec![0; length];
                        file.read_exact(&mut data)?;
                        Ok(MovieInput {
                            allows_movie_cache: false,
                            bytes: data.into(),
                            loader_request: None,
                            abort: None,
                            reservation,
                        })
                    })();
                    let _ = send.send(result);
                }
            });
            receive.await.map_err(io::Error::other)?
        }
    }
}
