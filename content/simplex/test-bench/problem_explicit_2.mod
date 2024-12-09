/* Problem 2: minimization */
var x1 >= 0;
var x2 >= 0;

minimize z: 31 * x1 +  51 * x2;

subject to
  c1: 1 * x1 +  21 * x2 <= 12;
  c2: 41 * x1 +  1 * x2 <= 16;
  c3: 1 * x1 +  1 * x2 <= 8;

end;
