use nanoodle::materialize;
use serde_json::json;

#[test]
fn saved_editor_example_resolves_to_its_public_url() {
    let graph = materialize(&json!({
        "nodes": [
            {"id":"mesh","type":"mupload","fields":{"model":"examples/product-shot/sample.glb"}},
            {"id":"bad","type":"upload","fields":{"image":"examples/../outside.png"}}
        ],
        "links": []
    })).unwrap();
    assert_eq!(graph.nodes[0].fields["model"], "https://nanoodle.com/examples/product-shot/sample.glb");
    assert_eq!(graph.nodes[1].fields["image"], "");
    assert_eq!(graph.warnings.len(), 1);
}
