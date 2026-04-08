import attrs
import jax

from jarp import tree


@tree.define(pytree="none")
class Example:
    data: jax.Array = tree.array(default=[1, 2])
    auto_value: object = tree.auto(default="auto")
    meta: str = tree.static(default="tag")


def test_field_specifier_metadata_matches_tree_roles() -> None:
    fields = attrs.fields_dict(Example)
    assert "static" not in fields["data"].metadata
    assert fields["auto_value"].metadata["static"] is tree.FieldType.AUTO
    assert tree.FieldType(fields["meta"].metadata["static"]) is tree.FieldType.META


def test_array_defaults_are_normalized_and_field_types_coerce_inputs() -> None:
    meta_flag = True
    assert isinstance(Example().data, jax.Array)
    assert Example().data.tolist() == [1, 2]
    assert tree.FieldType("auto") is tree.FieldType.AUTO
    assert tree.FieldType(meta_flag) is tree.FieldType.META
    assert bool(tree.FieldType.META)
    assert not bool(tree.FieldType.DATA)
