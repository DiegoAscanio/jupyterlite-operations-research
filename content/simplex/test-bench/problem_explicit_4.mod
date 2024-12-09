/* Problem 4: maximization */
var x1 >= 0;
var x2 >= 0;

maximize z: 21 * x1 +  41 * x2;

subject to
  c1: 1 * x1 +  1 * x2 >= 5;
  c2: 21 * x1 +  1 * x2 >= 8;
  c3: 1 * x1 +  31 * x2 >= 10;

end;
