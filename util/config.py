import os

device = 'cuda'  # Должно работать и с device = 'cpu', но будет обучаться дико долго и не факт что запустится
debug = True  # Писать ли картинки в тензорборду часто?
num_workers = 0  # Чисо потоков. На винде не работает. На убунте можно поставить  os.cpu_count()
inp_shape = (128, 128)  # Размер воходного изображения
discr_out = (1, 14, 14)  # Размер выхода дискриминатора
# По идее, его можно высчитать, но легче всего просто запустить tests.py и посмотреть что написано в предпоследней строке. Если выводит torch.Size([1, 1, 14, 14]), то пишем (1,14,14)

AVAIL_GPUS = 1 if device == 'cuda' else None  # не трогаем
BATCH_SIZE = 256  # размер батча. Не влзеает в память-уменьшаем, отим лучше резы - увеличиваем

X_PATH = "data_dir/data"
Y_PATH = "data_dir/data_out"
