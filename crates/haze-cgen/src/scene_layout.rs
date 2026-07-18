use crate::scene::{NodeId, NodeKind, SceneDocument, SceneError, SceneNode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PixelRect {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
}

impl PixelRect {
    pub(crate) fn canvas(width: u32, height: u32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        let left = self.x.max(other.x);
        let top = self.y.max(other.y);
        let right = (self.x + self.width).min(other.x + other.width);
        let bottom = (self.y + self.height).min(other.y + other.height);
        if right <= left || bottom <= top {
            return None;
        }
        Some(Self {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSceneNode<'a> {
    pub(crate) id: &'a NodeId,
    pub(crate) kind: &'a NodeKind,
    pub(crate) rect: PixelRect,
    pub(crate) clip: Option<PixelRect>,
    pub(crate) opacity: f32,
    pub(crate) z_index: i64,
    pub(crate) tree_order: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSceneLayout<'a> {
    pub(crate) canvas: PixelRect,
    pub(crate) nodes: Vec<ResolvedSceneNode<'a>>,
}

impl<'a> ResolvedSceneLayout<'a> {
    pub(crate) fn resolve(
        document: &'a SceneDocument,
        width: u32,
        height: u32,
    ) -> Result<Self, SceneError> {
        document.validate()?;
        let canvas = PixelRect::canvas(width.max(1), height.max(1));
        let mut nodes = Vec::new();
        let mut tree_order = 0usize;
        resolve_node(
            &document.root,
            canvas,
            Some(canvas),
            1.0,
            0,
            &mut tree_order,
            &mut nodes,
        );
        nodes.sort_by_key(|node| (node.z_index, node.tree_order));
        Ok(Self { canvas, nodes })
    }

    pub(crate) fn drawable_nodes(&self) -> impl Iterator<Item = &ResolvedSceneNode<'a>> {
        self.nodes.iter().filter(|node| {
            !matches!(node.kind, NodeKind::Group(_))
                && node.rect.width > 0.0
                && node.rect.height > 0.0
                && node.opacity > 0.0
                && node
                    .clip
                    .is_none_or(|clip| clip.width > 0.0 && clip.height > 0.0)
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_node<'a>(
    node: &'a SceneNode,
    parent: PixelRect,
    inherited_clip: Option<PixelRect>,
    inherited_opacity: f32,
    inherited_z: i64,
    tree_order: &mut usize,
    nodes: &mut Vec<ResolvedSceneNode<'a>>,
) {
    if !node.enabled {
        return;
    }
    let rect = resolve_rect(parent, node);
    let opacity = (inherited_opacity * node.transform.opacity).clamp(0.0, 1.0);
    let z_index = inherited_z.saturating_add(i64::from(node.transform.z_index));
    let order = *tree_order;
    *tree_order = tree_order.saturating_add(1);
    nodes.push(ResolvedSceneNode {
        id: &node.id,
        kind: &node.kind,
        rect,
        clip: inherited_clip,
        opacity,
        z_index,
        tree_order: order,
    });

    let child_clip = if node.transform.clip_children {
        match inherited_clip {
            Some(clip) => clip.intersection(rect),
            None => Some(rect),
        }
    } else {
        inherited_clip
    };
    if node.transform.clip_children && child_clip.is_none() {
        return;
    }
    for child in &node.children {
        resolve_node(child, rect, child_clip, opacity, z_index, tree_order, nodes);
    }
}

fn resolve_rect(parent: PixelRect, node: &SceneNode) -> PixelRect {
    let transform = node.transform;
    let left = parent.x + parent.width * transform.anchors.left + transform.x;
    let top = parent.y + parent.height * transform.anchors.top + transform.y;
    let right = parent.x + parent.width * transform.anchors.right + transform.x + transform.width;
    let bottom =
        parent.y + parent.height * transform.anchors.bottom + transform.y + transform.height;
    PixelRect {
        x: left,
        y: top,
        width: (right - left).max(0.0),
        height: (bottom - top).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::{
        protected_default_scene, Anchors, ColorNode, ColorSource, GroupNode, NodeId, RectTransform,
        RgbaColor, SceneId, SceneNode,
    };

    use super::*;

    fn node(id: &str, transform: RectTransform, children: Vec<SceneNode>) -> SceneNode {
        SceneNode {
            id: NodeId::new(id).expect("valid ID"),
            name: id.to_string(),
            enabled: true,
            transform,
            kind: NodeKind::Group(GroupNode::default()),
            children,
        }
    }

    fn full_canvas_transform() -> RectTransform {
        RectTransform {
            anchors: Anchors {
                left: 0.0,
                top: 0.0,
                right: 1.0,
                bottom: 1.0,
            },
            ..RectTransform::default()
        }
    }

    #[test]
    fn protected_layout_snapshots_scale_across_sd_hd_and_wide_canvases() {
        let crawl = protected_default_scene(crate::scene::ProtectedSceneKind::StandardCrawl);
        for (width, height, expected_top) in [
            (720, 480, 360.0),
            (1_280, 720, 600.0),
            (1_920, 1_080, 960.0),
            (1_920, 800, 680.0),
        ] {
            let layout = ResolvedSceneLayout::resolve(&crawl, width, height).expect("layout");
            let background = layout
                .nodes
                .iter()
                .find(|node| node.id.as_str() == "crawl_background")
                .expect("crawl background");
            assert_eq!(background.rect.x, 0.0);
            assert_eq!(background.rect.y, expected_top);
            assert_eq!(background.rect.width, width as f32);
            assert_eq!(background.rect.height, 120.0);
        }
    }

    #[test]
    fn child_layout_opacity_clipping_and_z_are_hierarchical() {
        let child_transform = RectTransform {
            anchors: Anchors {
                left: 0.0,
                top: 0.0,
                right: 1.0,
                bottom: 1.0,
            },
            x: 10.0,
            y: 10.0,
            width: -20.0,
            height: -20.0,
            z_index: 2,
            opacity: 0.5,
            clip_children: false,
        };
        let mut parent_transform = child_transform;
        parent_transform.anchors.right = 0.5;
        parent_transform.anchors.bottom = 0.5;
        parent_transform.width = -20.0;
        parent_transform.height = -20.0;
        parent_transform.z_index = 3;
        parent_transform.opacity = 0.5;
        parent_transform.clip_children = true;
        let document = SceneDocument {
            schema_version: 1,
            id: SceneId::new("custom").expect("valid ID"),
            name: "custom".to_string(),
            root: node(
                "root",
                full_canvas_transform(),
                vec![node(
                    "parent",
                    parent_transform,
                    vec![node("child", child_transform, Vec::new())],
                )],
            ),
        };
        let layout = ResolvedSceneLayout::resolve(&document, 1_000, 500).expect("layout");
        let parent = layout
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "parent")
            .expect("parent");
        let child = layout
            .nodes
            .iter()
            .find(|node| node.id.as_str() == "child")
            .expect("child");
        assert_eq!(
            parent.rect,
            PixelRect {
                x: 10.0,
                y: 10.0,
                width: 480.0,
                height: 230.0
            }
        );
        assert_eq!(
            child.rect,
            PixelRect {
                x: 20.0,
                y: 20.0,
                width: 460.0,
                height: 210.0
            }
        );
        assert_eq!(child.clip, Some(parent.rect));
        assert_eq!(child.opacity, 0.25);
        assert_eq!(child.z_index, 5);
    }

    #[test]
    fn equal_z_uses_stable_tree_order_and_disabled_subtrees_are_absent() {
        let transform = RectTransform {
            z_index: 5,
            ..RectTransform::default()
        };
        let mut hidden = node(
            "hidden",
            transform,
            vec![node("hidden_child", transform, Vec::new())],
        );
        hidden.enabled = false;
        let document = SceneDocument {
            schema_version: 1,
            id: SceneId::new("ordering").expect("valid ID"),
            name: "ordering".to_string(),
            root: node(
                "root",
                full_canvas_transform(),
                vec![
                    node("first", transform, Vec::new()),
                    node("second", transform, Vec::new()),
                    hidden,
                ],
            ),
        };
        let layout = ResolvedSceneLayout::resolve(&document, 720, 480).expect("layout");
        let ordered = layout
            .nodes
            .iter()
            .filter(|node| node.z_index == 5)
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec!["first", "second"]);
        assert!(!layout
            .nodes
            .iter()
            .any(|node| node.id.as_str().starts_with("hidden")));
    }

    #[test]
    fn drawable_filter_skips_groups_and_zero_area_nodes() {
        let color = SceneNode {
            id: NodeId::new("color").expect("valid ID"),
            name: "color".to_string(),
            enabled: true,
            transform: RectTransform {
                anchors: Anchors {
                    left: 0.0,
                    top: 0.0,
                    right: 1.0,
                    bottom: 1.0,
                },
                ..RectTransform::default()
            },
            kind: NodeKind::Color(ColorNode {
                source: ColorSource::Static(RgbaColor::new(255, 0, 0, 255)),
            }),
            children: Vec::new(),
        };
        let document = SceneDocument {
            schema_version: 1,
            id: SceneId::new("drawables").expect("valid ID"),
            name: "drawables".to_string(),
            root: node("root", full_canvas_transform(), vec![color]),
        };
        let layout = ResolvedSceneLayout::resolve(&document, 720, 480).expect("layout");
        assert_eq!(layout.drawable_nodes().count(), 1);
    }
}
