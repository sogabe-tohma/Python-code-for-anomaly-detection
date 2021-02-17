# -*- coding: utf-8 -*-
from future import standard_library
standard_library.install_aliases()
import threading
from PIL import Image


class _ImageThread(threading.Thread):

    def __init__(self, filenames, results, color="RGB"):
        super(_ImageThread, self).__init__()

        self._filenames = filenames
        self._results = results
        color_key = {'GRAY': 'L', 'RGB': 'RGB'}
        self._color = color_key[color]

    def run(self):
        for filename in self._filenames:
            img = Image.open(filename)
            # Call load() method explicitly to let PIL to close file
            img.load()
            img = img.convert(self._color)
            self._results.append(img)


class ImageLoader(object):
    """ImageLoader is a generator that yields images in batches.
    By inputting list of image path, ImageLoader load images and
    yields according to number of batch size.

    Args:
        batches (list): List of image path.
        color (str): Color Space of Input Image.

    Example:
        >>> batches = [[('/data/file1.jpg', '/data/file2.jpg')], [('/data/file3.jpg', '/data/file4.jpg')] ]
        >>> loader = ImageLoader(batches)
        >>> for i, (x) in enumerate(dist.batch(2)):
        ...    print 'Batch', i
    """

    NUMTHREADS = 8  # Number of image reader thread

    def __init__(self, batches, color="RGB"):
        self._batches = batches
        self._images = None
        self._loaded = threading.Event()
        self._loading = False
        self._color = color

    def wait_images(self):
        for imgs in self._imageloader(self._batches):
            yield imgs

    def _imageloader(self, batches):
        if not batches:
            return

        i = 0
        th, results = self._readimgs(batches[0])
        while i < len(batches):
            th.join()
            cur = results
            if (i + 1) < len(batches):
                th, results = self._readimgs(batches[i + 1])
            yield cur
            i += 1

    def _readimgs(self, filenames):
        args = []
        # Split filenames
        div, mod = divmod(len(filenames), self.NUMTHREADS)

        if div == 0:
            args = [filenames]
        else:
            for i in range(0, div * self.NUMTHREADS, div):
                args.append(filenames[i:i + div])

            if mod:
                args[-1].extend(filenames[-1 * mod:])

        # Prepare threads
        results = [[] for _ in args]
        threads = [_ImageThread(arg, result, self._color)
                   for result, arg in zip(results, args)]

        # start threads
        for thread in threads:
            thread.start()

        ret = []

        def _wait():
            # wait for all threads to terminate
            for thread in threads:
                thread.join()

            # flatten result
            for r in results:
                ret.extend(r)

        th = threading.Thread(target=_wait)
        th.start()
        return th, ret
