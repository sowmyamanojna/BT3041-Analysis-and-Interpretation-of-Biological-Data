[x,y] = meshgrid(-2:.1:2);

z1 = g(x+y-1.5);
z2 = g(x+y-0.5);

z = g(z2-z1-0.5);
mesh(x,y,z)