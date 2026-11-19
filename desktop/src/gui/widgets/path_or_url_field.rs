use crate::gui::{FilePicker, LocalizableText, text};
use egui::{TextEdit, Ui};
use ruffle_frontend_utils::content::ContentDescriptor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use unic_langid::LanguageIdentifier;
use url::Url;

#[derive(Default)]
struct PathOrUrlFieldValue {
    content_descriptor: Option<ContentDescriptor>,
    representation: String,
    changed: bool,
}

impl PathOrUrlFieldValue {
    fn new(value: ContentDescriptor) -> Self {
        let url = &value.url;
        Self {
            representation: if url.scheme() == "file"
                && let Ok(path) = url.to_file_path()
            {
                path.to_string_lossy().to_string()
            } else {
                url.to_string()
            },
            content_descriptor: Some(value),
            changed: false,
        }
    }

    fn from_path(path: PathBuf) -> Self {
        Self {
            representation: Self::path_to_representation(&path),
            content_descriptor: ContentDescriptor::new_local(&path, None),
            changed: false,
        }
    }

    fn from_string(string: String) -> Self {
        let path = Path::new(&string);
        if path.is_file() {
            Self::from_path(path.to_path_buf())
        } else {
            Self {
                content_descriptor: Url::parse(&string).map(ContentDescriptor::new_remote).ok(),
                representation: string,
                changed: false,
            }
        }
    }

    /// When the user picked a directory, we want to show the directory, but
    /// encode the real path we want to play.
    fn from_picked_directory(directory: PathBuf, content: PathBuf) -> Self {
        Self {
            representation: Self::path_to_representation(&directory),
            content_descriptor: ContentDescriptor::new_local(&content, Some(directory)),
            changed: false,
        }
    }

    fn replace(&mut self, mut next: Self) {
        next.changed = self.changed || self.content_descriptor != next.content_descriptor;
        *self = next;
    }

    fn path_to_representation(path: &Path) -> String {
        if let Some(name) = path.file_name() {
            name.to_string_lossy().into_owned()
        } else {
            path.to_string_lossy().into_owned()
        }
    }
}

pub struct PathOrUrlField {
    picker: FilePicker,
    value: Arc<Mutex<PathOrUrlFieldValue>>,
    hint: LocalizableText,
}

impl PathOrUrlField {
    pub fn new(
        default_content: Option<ContentDescriptor>,
        hint: LocalizableText,
        picker: FilePicker,
    ) -> Self {
        Self {
            picker,
            value: Arc::new(Mutex::new(
                default_content
                    .map(PathOrUrlFieldValue::new)
                    .unwrap_or_default(),
            )),
            hint,
        }
    }

    fn lock_value(value: &Arc<Mutex<PathOrUrlFieldValue>>) -> MutexGuard<'_, PathOrUrlFieldValue> {
        value.lock().expect("Non-poisoned value")
    }

    pub fn ui(&mut self, locale: &LanguageIdentifier, ui: &mut Ui) -> bool {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let directory = || {
                Self::lock_value(&self.value)
                    .content_descriptor
                    .as_ref()
                    .map(|desc| &desc.url)
                    .filter(|url| url.scheme() == "file")
                    .and_then(|url| url.to_file_path().ok())
                    .map(|mut path| {
                        path.pop();
                        path
                    })
            };

            if ui
                .button(text(locale, "path-or-url-field-open-directory"))
                .clicked()
            {
                let dir = directory();
                let ctx = ui.ctx().clone();
                let value = self.value.clone();
                let picker = self.picker.clone();
                tokio::spawn(async move {
                    if let Some((directory, content)) =
                        picker.pick_ruffle_directory_and_content(dir).await
                    {
                        Self::lock_value(&value).replace(
                            PathOrUrlFieldValue::from_picked_directory(directory, content),
                        );
                        ctx.request_repaint();
                    }
                });
            }

            if ui
                .button(text(locale, "path-or-url-field-open-file"))
                .clicked()
            {
                let dir = directory();
                let ctx = ui.ctx().clone();
                let value = self.value.clone();
                let picker = self.picker.clone();
                tokio::spawn(async move {
                    if let Some(path) = picker.pick_ruffle_file(dir).await {
                        Self::lock_value(&value).replace(PathOrUrlFieldValue::from_path(path));
                        ctx.request_repaint();
                    }
                });
            }

            let mut value = Self::lock_value(&self.value);
            let color = value
                .content_descriptor
                .is_none()
                .then(|| ui.style().visuals.error_fg_color);
            let response = ui.add_sized(
                ui.available_size(),
                TextEdit::singleline(&mut value.representation)
                    .hint_text(self.hint.localize(locale))
                    .text_color_opt(color),
            );
            if response.changed() {
                let typed = std::mem::take(&mut value.representation);
                value.replace(PathOrUrlFieldValue::from_string(typed));
            }
            std::mem::take(&mut value.changed)
        })
        .inner
    }

    pub fn is_valid(&self) -> bool {
        Self::lock_value(&self.value).content_descriptor.is_some()
    }

    pub fn result(&self) -> Option<ContentDescriptor> {
        Self::lock_value(&self.value).content_descriptor.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picked_location_is_delivered_once_and_preserves_the_directory() {
        let mut field = PathOrUrlFieldValue::from_string("https://example.org/first.swf".into());
        let directory = std::env::temp_dir().join("bookmark");
        let content = directory.join("second.swf");
        assert!(!field.changed);
        field.replace(PathOrUrlFieldValue::from_picked_directory(
            directory.clone(),
            content.clone(),
        ));
        field.replace(PathOrUrlFieldValue::from_picked_directory(
            directory.clone(),
            content.clone(),
        ));
        assert!(std::mem::take(&mut field.changed));
        assert!(!field.changed);
        let expected = ContentDescriptor::new_local(&content, Some(directory));
        assert_eq!(field.content_descriptor, expected);
        field.replace(PathOrUrlFieldValue::new(expected.unwrap()));
        assert!(!field.changed);
    }
}
