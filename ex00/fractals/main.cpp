#include <iostream>
#include <cmath>
#include <string>
#include <mgl2/mgl.h>


class Point {
  private:
    double x;
    double y;
    Point *next;

  public:
    Point(double x, double y) : x(x), y(y), next(nullptr) {}
    Point(double x, double y, Point *next) : x(x), y(y), next(next) {}

    double dist(Point &a, Point &b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;

        return std::sqrt(dx*dx + dy*dy);
    }

    int countPoints() {
        int counter = 0;

        Point *iter = this;
        while (iter != nullptr) {
            counter++;
            iter = iter->next;
        }

        return counter;
    }

    double length() {
        Point *iter = this;

        double len = 0;
        while (iter->next != nullptr) {
            len += this->dist(*iter, *iter->next);
            iter=iter->next;
        }

        return len;
    }

    void toArrays(double* xcoords, double* ycoords) {
        Point *iter = this;
        int i = 0;

        while (iter != nullptr) {
            xcoords[i] = iter->x;
            ycoords[i] = iter->y;

            i++;
            iter = iter->next;
        }
    }

    void plot(const char *name) {
        int len = this->countPoints();
        double* xcoords = new double[len];
        double* ycoords = new double[len];
        this->toArrays(xcoords, ycoords);

        mglData datx, daty;
        datx.Link(xcoords, len);
        daty.Link(ycoords, len);
        mglGraph gr;
        gr.SetRanges(0,1,0,1);
        gr.Plot(datx, daty, "0");
        mglPoint pleftdown(0,0), prightup(1,1);
        gr.SetCutBox(pleftdown, prightup);
        gr.WriteFrame(name);
    }

    void fractalizeSegment() {
        Point *p = this->next;

        double xonethird = (2.0*this->x + p->x)/3.0;
        double yonethird = (2.0*this->y + p->y)/3.0;
        double xmid = (this->x + p->x)/2.0;
        double ymid = (this->y + p->y)/2.0;
        double xtwothird = (this->x + 2.0*p->x)/3.0;
        double ytwothird = (this->y + 2.0*p->y)/3.0;
        double xperp = yonethird - ytwothird;
        double yperp = xtwothird - xonethird;

        Point *p3 = new Point(xtwothird, ytwothird, p);
        Point *p2 = new Point(xmid + xperp*sqrt(3)/2, ymid + yperp*sqrt(3)/2, p3);
        Point *p1 = new Point(xonethird, yonethird, p2);
        this->next = p1;
    }

    void fractalize() {
        Point *iter = this;

        while (iter->next != nullptr) {
            Point *next = iter->next;
            iter->fractalizeSegment();
            iter = next;
        }
    }
};


int main() {
    Point *p2 = new Point(1,0);
    Point *p1 = new Point(0,0,p2);

    int nStages = 5;
    double *lengths = new double[nStages];

    for (int i = 0; i < nStages; ++i) {
        p1->fractalize();
        std::string name = std::string("bin/fractalize") + std::to_string(i+1) + std::string(".eps");
        const char *cname = name.c_str();
        p1->plot(cname);
        lengths[i] = p1->length();
    }

    mglData data;
    data.Link(lengths, nStages);
    mglGraph gr;
    gr.SetRanges(0,nStages,0,nStages);
    gr.Plot(data);
    gr.Axis();
    gr.Label('x',"Stage");
    gr.Label('y',"Length");
    gr.WriteFrame("bin/lengthsPlot.png");
}

