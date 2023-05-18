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