#include <SFML/Graphics.hpp>
#include <complex>

//fractals generally created by counting iterations of complex function until value reach threshold
using ComplexFunction = std::function<std::complex<double>(std::complex<double>)>;

//function to calculate a fractal
int count(std::complex<double> c, int maxIter, ComplexFunction func) {
    std::complex<double> z (0,0);
    for (int i = 0; i < maxIter; i++){
        if (abs(z) > 4){
            return i;
        }
        z=func(z)+c;
    }

    return maxIter;
}

int main() {

    //USER DEFINED:
    ComplexFunction fractal = [](std::complex<double> z) {
        //in this case, mandelbrot
        return z * z;
    };

    const int width = 800;
    const int height = 600;
    const int maxIter = 255;
    const double scale = 3.0 / height; // Adjust scale factor as needed

    //DO NOT CHANGE:
    sf::RenderWindow window(sf::VideoMode(width, height), "Mandelbrot Set");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                //compute:
                //re in [-width/2, width/2]*scale
                //im in [-height/2, height/2]*scale
                std::complex<double> c ((x - width / 2) * scale,(y - height / 2) * scale);
                int iter = count(c, maxIter, fractal);

                //plot:
                sf::Color color(255 * iter / maxIter, 255 * iter / maxIter, 255 * iter / maxIter);
                sf::Vertex point(sf::Vector2f(x, y), color);
                window.draw(&point, 1, sf::Points);
            }
        }

        window.display();
    }

    return 0;
}
