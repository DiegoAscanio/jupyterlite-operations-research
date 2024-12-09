/* Problem 10: mixed */
var x1 >= 0;
var x2 >= 0;

subject to
  c1: 1 * x1 +  1 * x2 <= 10;
  c2: 21 * x1 +  1 * x2 >= 15;
  c3: 1 * x1 +  1 * x2 = 8;

end;
