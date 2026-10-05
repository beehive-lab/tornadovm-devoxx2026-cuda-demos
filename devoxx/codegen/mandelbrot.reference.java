static void mandelbrot(int width, int height, int maxIterations, IntArray out) {
    for (@Parallel int y = 0; y < height; y++) {
        for (@Parallel int x = 0; x < width; x++) {
            float real = -2.2f + (x / (float) width) * 3.2f;
            float imag = -1.2f + (y / (float) height) * 2.4f;
            float zReal = 0.0f, zImag = 0.0f;
            int iter = 0;
            while (iter < maxIterations) {
                float zReal2 = zReal * zReal - zImag * zImag + real;
                float zImag2 = 2 * zReal * zImag + imag;
                zReal = zReal2;
                zImag = zImag2;
                if (zReal * zReal + zImag * zImag > 4.0f) {
                    break;
                }
                iter++;
            }
            out.set(y * width + x, iter);
        }
    }
}
