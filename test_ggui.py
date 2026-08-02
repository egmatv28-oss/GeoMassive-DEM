import taichi as ti

ti.init(arch=ti.vulkan)

w = ti.ui.Window('test', (128, 96), show_window=True)
c = w.get_canvas()
img = ti.Vector.field(3, ti.f32, (128, 96))


@ti.kernel
def fill():
    for X, Y in img:
        img[X, Y] = ti.Vector([1.0, 0.0, 0.0])


fill()
c.set_image(img)
w.show()
print('junction-based import OK')
