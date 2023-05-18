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

