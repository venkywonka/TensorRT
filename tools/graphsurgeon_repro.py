import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 5, 5))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
Z = gs.Variable(name="Z", dtype=np.float32, shape=(1, 3, 1, 1))
node1 = gs.Node(op="GlobalLpPool", name="op0", attrs={"p": 2}, inputs=[X], outputs=[Y])
node2 = gs.Node(op="Identity", name="op0", inputs=[Y], outputs=[Z])

# node1 and node2 have the same name, no warning.
graph = gs.Graph(nodes=[node1, node2], inputs=[X], outputs=[Z])
print(graph)
onnx.save(gs.export_onnx(graph), "two_nodes_same_name.onnx")



# Tensors A and B have the same name, we get a warning
A = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
B = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
node3 = gs.Node(op="Identity", name="op0", inputs=[A], outputs=[B])
graph2 = gs.Graph(nodes=[node3], inputs=[A], outputs=[B])
print(graph2)
onnx.save(gs.export_onnx(graph2), "two_tensors_same_name.onnx")


# 3 node with the same name, expected warning twice.
V = gs.Variable(name="V", dtype=np.float32, shape=(1, 3, 5, 5))
W = gs.Variable(name="W", dtype=np.float32, shape=(1, 3, 1, 1))
T = gs.Variable(name="T", dtype=np.float32, shape=(1, 3, 1, 1))
U = gs.Variable(name="U", dtype=np.float32, shape=(1, 3, 1, 1))

node1 = gs.Node(op="GlobalLpPool", name="op0", attrs={"p": 2}, inputs=[V], outputs=[W])
node2 = gs.Node(op="Identity", name="op0", inputs=[W], outputs=[T])
node3 = gs.Node(op="Identity", name="op0", inputs=[T], outputs=[U])

graph3 = gs.Graph(nodes=[node1, node2, node3], inputs=[V], outputs=[U])
print(graph3)
onnx.save(gs.export_onnx(graph3), "three_nodes_same_name.onnx")

# 4 nodes, with 2 nodes same name and 2 nodes empty name, expected warning only once
S = gs.Variable(name="S", dtype=np.float32, shape=(1, 3, 5, 5))
T = gs.Variable(name="T", dtype=np.float32, shape=(1, 3, 1, 1))
U = gs.Variable(name="U", dtype=np.float32, shape=(1, 3, 1, 1))
V = gs.Variable(name="V", dtype=np.float32, shape=(1, 3, 5, 5))
W = gs.Variable(name="W", dtype=np.float32, shape=(1, 3, 1, 1))
X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 1, 1))

node1 = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[S], outputs=[T])
node2 = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[V], outputs=[W])
node3 = gs.Node(op="Identity", name="op0", inputs=[W], outputs=[U])
node4 = gs.Node(op="Identity", name="op0", inputs=[T], outputs=[X])

graph4 = gs.Graph(nodes=[node1, node2, node3, node4], inputs=[S, V], outputs=[U, X])
print(graph4)
onnx.save(gs.export_onnx(graph4), "two_nodes_same_name_two_empty.onnx")

# test to check if `onnx_graphsurgeon_node_0` is duplicated in layer() method.
X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 5, 5))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
Z = gs.Variable(name="Z", dtype=np.float32, shape=(1, 3, 1, 1))
T = gs.Variable(name="T", dtype=np.float32, shape=(1, 3, 1, 1))
U = gs.Variable(name="U", dtype=np.float32, shape=(1, 3, 1, 1))
W = gs.Variable(name="W", dtype=np.float32, shape=(1, 3, 1, 1))

node1 = gs.Node(op="GlobalLpPool", name="onnx_graphsurgeon_node_0", attrs={"p": 2}, inputs=[X], outputs=[Y])
node2 = gs.Node(op="Identity", name="onnx_graphsurgeon_node_1", inputs=[Y], outputs=[Z])

graph = gs.Graph(nodes=[node1, node2], inputs=[X], outputs=[Z])
graph.layer(inputs=[Z], outputs=[T], op="Identity")
graph.layer(inputs=[T], outputs=[U], op="identity")
print(graph)
onnx.save(gs.export_onnx(graph), "duplicating_via_layer.onnx")

node1 = gs.Node(op="Identity", name="onnx_graphsurgeon_node_0") # default name already used
node2 = gs.Node(op="Identity", name="onnx_graphsurgeon_node_1") # default name already used
graph = gs.Graph(nodes=[node1, node2])
graph.layer(op="Identity")
assert graph.nodes[-1].name == "onnx_graphsurgeon_node_2"
graph.layer(op="Identity")
assert graph.nodes[-1].name == "onnx_graphsurgeon_node_3"