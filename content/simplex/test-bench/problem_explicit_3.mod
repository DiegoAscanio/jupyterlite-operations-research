/* Problem 3: minimization */
var x1 >= 0;
var x2 >= 0;

minimize z: 41 * x1 +  1 * x2;

subject to
  c1: 1 * x1 +  1 * x2 <= 20;
  c2: 21 * x1 +  31 * x2 <= 30;
  c3: 31 * x1 +  1 * x2 <= 25;

end;
