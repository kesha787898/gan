pip install -r requirements.txt установит зависимости(возможно, я какую-нибудь пропустил)

**config.py**  
Заполняем согласно комментам

**change.py**  
Этот скрипт подготовит изображения для обучения. Запускать нужно только 1 раз для датасета, а не перед каждым обучением

**train.py**   
Просто запускаем, можно поменять максимальное число эпох. Дальше запускаем тензорборд

запускаем tensorboard --logdir tb_logs и смотрим картинки(сверху Images).В центре это то, что сгенерировалось. Как только нас устроило,
берем в папке chkpts чекпоинт и радуемся жизни.  
Пример того, как получить предсказания с помощью чекпоинта eval.py

Если хотите поэксперементировать с сеткой, то меняете в Gan.py ти строчки :  
self.generator = UnetGenerator(1, 3, num_downs=3, ngf=12)  #менять num_downs и ngf   
self.discriminator =NLayerDiscriminator(4, ndf=36, n_layers=3)# менять ndf, n_layers  
Лучше менять  ngf и ndf
