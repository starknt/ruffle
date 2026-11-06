//! A trusted frontend can load a small ActionScript adapter next to a movie's definitions.
//! The handle roots the adapter class and its private child domain in one player.
use crate::Player;
use crate::avm2::{Activation, Avm2, Domain, FunctionArgs, QName, Value};
use crate::display_object::TDisplayObject;
use crate::string::AvmString;
use gc_arena::{DynamicRoot, Gc, Rootable};

pub struct HostAdapter(DynamicRoot<Rootable![(Domain<'_>, Value<'_>)]>);

impl HostAdapter {
    /// Inspect the decompressed tag bytes of the movie that actually defined a class.
    pub fn class_movies(player: &mut Player, class: &str) -> Vec<std::sync::Arc<Vec<u8>>> {
        player.mutate_with_update_context(|ctx| {
            let name = QName::from_qualified_name(AvmString::new_utf8(ctx.gc(), class), ctx);
            let mut owners = Vec::new();
            for movie in ctx.library.known_movies() {
                let Some(class) = ctx
                    .library
                    .library_for_movie(movie)
                    .and_then(|lib| lib.try_avm2_domain())
                    .and_then(|domain| domain.classes().get(name).copied())
                else {
                    continue;
                };
                if let Some(init) = class.instance_init() {
                    let bytes = init.owner_movie().shared_data();
                    if !owners
                        .iter()
                        .any(|known| std::sync::Arc::ptr_eq(known, &bytes))
                    {
                        owners.push(bytes);
                    }
                }
            }
            owners
        })
    }

    /// Locate the domain that declares `anchor` and load trusted ABC into a private child.
    /// Returns `None` while the movie containing the anchor has not loaded.
    pub fn load(
        player: &mut Player,
        anchor: &str,
        class: &str,
        abc: &[u8],
    ) -> Result<Option<Self>, &'static str> {
        player.mutate_with_update_context(|ctx| {
            let anchor = AvmString::new_utf8(ctx.gc(), anchor);
            let anchor = QName::from_qualified_name(anchor, ctx);
            let found = ctx.library.known_movies().find_map(|movie| {
                let domain = ctx.library.library_for_movie(movie)?.try_avm2_domain()?;
                domain
                    .classes()
                    .contains_key(anchor)
                    .then_some((movie, domain))
            });
            let Some((movie, parent)) = found else {
                return Ok(None);
            };
            let class = QName::from_qualified_name(AvmString::new_utf8(ctx.gc(), class), ctx);
            if parent.has_definition(class) {
                return Err("Host adapter name is already defined by the movie");
            }
            let domain = Domain::movie_domain(ctx, parent);
            Avm2::do_abc(
                ctx,
                abc,
                None,
                swf::DoAbc2Flag::LAZY_INITIALIZE,
                domain,
                movie,
            )
            .map_err(|_| "Invalid host adapter bytecode")?;
            if !domain.classes().contains_key(class) {
                return Err("Host adapter bytecode did not declare the requested class");
            }
            let mut activation = Activation::from_domain(ctx, domain);
            let class = domain
                .get_defined_value(&mut activation, class)
                .map_err(|_| "Host adapter class was not defined")?;
            Ok(Some(Self(
                ctx.dynamic_root
                    .stash(ctx.gc(), Gc::new(ctx.gc(), (domain, class))),
            )))
        })
    }

    /// Invoke a static adapter method with the stage and a string. AVM objects never escape the player.
    pub fn call(
        &self,
        player: &mut Player,
        method: &str,
        input: &str,
    ) -> Result<String, &'static str> {
        player.mutate_with_update_context(|ctx| {
            let (domain, class) = *ctx
                .dynamic_root
                .try_fetch(&self.0)
                .map_err(|_| "Host adapter belongs to another player")?;
            let stage = ctx.stage.object2().ok_or("Movie stage unavailable")?;
            let args = [stage.into(), AvmString::new_utf8(ctx.gc(), input).into()];
            let name = AvmString::new_utf8(ctx.gc(), method);
            let mut activation = Activation::from_domain(ctx, domain);
            let value = class
                .call_public_property(name, FunctionArgs::from_slice(&args), &mut activation)
                .map_err(|_| "ActionScript host adapter call failed")?;
            match value {
                Value::String(s) => Ok(s.to_string()),
                _ => Err("Host adapter returned a non-string value"),
            }
        })
    }
}
