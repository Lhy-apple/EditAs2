/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:58:25 GMT 2023
 */

package org.apache.commons.math3.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.fraction.Fraction;
import org.apache.commons.math3.fraction.FractionField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Fraction_ESTest extends Fraction_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      Fraction fraction1 = fraction0.multiply(5594265);
      assertEquals(3.356559E8, fraction1.percentageValue(), 0.01);
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      int int0 = fraction0.getDenominator();
      assertEquals(5, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.divide((-3366));
      boolean boolean0 = fraction0.equals(fraction1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.ZERO.subtract(fraction0);
      assertEquals((-1), fraction1.getNumerator());
      assertEquals((-0.25), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      FractionField fractionField0 = fraction0.TWO_QUARTERS.getField();
      assertNotNull(fractionField0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0.039504883579866146, 0.039504883579866146, (-1));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert 0.04 to fraction after -1 iterations
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      long long0 = fraction0.longValue();
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = new Fraction((-775.3787878351836));
      assertEquals((-51175), fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      Fraction fraction1 = fraction0.subtract(22);
      assertEquals((-106), fraction1.getNumerator());
      assertEquals((-21.2), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = new Fraction((-764));
      int int0 = fraction0.intValue();
      assertEquals((-764), int0);
      assertEquals((-764), fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      fraction0.hashCode();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      int int0 = fraction0.getNumerator();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = new Fraction((-324.3662582), (-1));
      float float0 = fraction0.floatValue();
      assertEquals((-325.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      double double0 = fraction0.percentageValue();
      assertEquals(50.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(1.7976931348623157E308, 339.389, Integer.MAX_VALUE);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 179,769,313,486,231,570,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000,000 to fraction (9,223,372,036,854,775,807/1)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = new Fraction(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE);
      assertEquals(2.147483647E9, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-2147483623), (-2147483623), (-2147483623));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -2,147,483,623 to fraction (-9,223,372,034,707,292,184/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) (-3472), (-3472));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -3,472 to fraction (3,473/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(629, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 629/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.MINUS_ONE.abs();
      assertEquals(1, fraction1.getNumerator());
      assertEquals(100.0, fraction1.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_QUARTERS;
      Fraction fraction1 = fraction0.TWO_FIFTHS.abs();
      int int0 = fraction1.compareTo(fraction0);
      assertEquals(0.4F, fraction1.floatValue(), 0.01F);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Fraction fraction0 = new Fraction((-764));
      Fraction fraction1 = fraction0.negate();
      int int0 = fraction1.compareTo(fraction0);
      assertEquals(764, fraction1.getNumerator());
      assertEquals(1, int0);
      assertEquals((-76400.0), fraction0.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      boolean boolean0 = fraction0.equals(fraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      boolean boolean0 = fraction0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      Fraction fraction1 = fraction0.divide((-3366));
      boolean boolean0 = fraction0.equals(fraction1);
      assertEquals((-1), fraction1.getNumerator());
      assertFalse(boolean0);
      assertEquals((-1.18835414E-4F), fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_FIFTHS;
      Fraction fraction1 = fraction0.divide(1885);
      boolean boolean0 = fraction0.equals(fraction1);
      assertEquals(9425, fraction1.getDenominator());
      assertEquals(0.021220159151193636, fraction1.percentageValue(), 0.01);
      assertFalse(fraction1.equals((Object)fraction0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(567, 567);
      Fraction fraction1 = fraction0.TWO_QUARTERS.reciprocal();
      Fraction fraction2 = fraction1.add(2147483646);
      // Undeclared exception!
      try { 
        fraction2.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/1, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      // Undeclared exception!
      try { 
        fraction0.ONE_HALF.subtract((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = new Fraction(183, 183);
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO_THIRDS.add(fraction1);
      assertEquals(1, fraction0.getDenominator());
      assertEquals(100.0, fraction0.percentageValue(), 0.01);
      assertEquals(2, fraction2.getNumerator());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = Fraction.ONE_FIFTH;
      Fraction fraction2 = fraction0.add(fraction1);
      assertEquals(0.95, fraction2.doubleValue(), 0.01);
      assertEquals(19, fraction2.getNumerator());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = fraction0.ONE_FIFTH.subtract(fraction0);
      assertEquals((-3), fraction1.getNumerator());
      assertEquals((-0.3), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = Fraction.ONE_FIFTH;
      Fraction fraction2 = fraction0.ONE_FIFTH.subtract(fraction1);
      assertEquals(1, fraction2.getDenominator());
      assertEquals(0.0, fraction2.percentageValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = fraction0.add(fraction0);
      assertEquals(1.5F, fraction1.floatValue(), 0.01F);
      assertEquals(3, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(553, (-2146453185));
      Fraction fraction1 = Fraction.ONE_QUARTER;
      Fraction fraction2 = fraction0.TWO_FIFTHS.divide(fraction1);
      // Undeclared exception!
      try { 
        fraction2.add(fraction0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: 3,434,324,543
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      // Undeclared exception!
      try { 
        fraction0.TWO_THIRDS.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = fraction0.ZERO.multiply(fraction0);
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO_FIFTHS.multiply(fraction1);
      assertEquals(0, fraction2.getNumerator());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.FOUR_FIFTHS;
      // Undeclared exception!
      try { 
        fraction0.MINUS_ONE.divide((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.MINUS_ONE.divide(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, (-1182));
      assertEquals(0, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      assertEquals(1, fraction0.getNumerator());
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(4897, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 4,897/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-1193));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-1,193, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      String string0 = fraction0.toString();
      assertEquals("1 / 2", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      String string0 = fraction0.toString();
      assertEquals("0", string0);
  }
}
