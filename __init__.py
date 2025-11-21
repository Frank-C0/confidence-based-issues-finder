"""
Example operators.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
# import asyncio

# import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

class CustomViewExample(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="example_custom_view",
            label="Examples: Custom View",
        )

    def execute(self, ctx):
        return {}

    def resolve_input(self, ctx):
        inputs = types.Object()
        component = types.View(
            label="My custom component", component="ExampleCustomView"
        )
        inputs.define_property(
            "component",
            types.String(),
            view=component,
            invalid=True,
            error_message="Custom error message",
        )
        return types.Property(inputs)



def register(p):
    p.register(CustomViewExample)
