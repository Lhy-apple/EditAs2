/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:22:04 GMT 2023
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
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.ONE_THIRD.add(fraction0);
      assertEquals(0.5833333F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      double double0 = fraction0.percentageValue();
      assertEquals(100.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      int int0 = fraction0.getDenominator();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_QUARTERS;
      Fraction fraction1 = fraction0.add(3);
      assertEquals(4, fraction1.getDenominator());
      assertEquals(3.75F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_HALF;
      Fraction fraction1 = fraction0.divide(1);
      assertTrue(fraction1.equals((Object)fraction0));
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertEquals(1, fraction1.getDenominator());
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = new Fraction(2906);
      assertEquals(2906, fraction0.getNumerator());
      assertEquals((byte)90, fraction0.byteValue());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      FractionField fractionField0 = fraction0.TWO_FIFTHS.getField();
      assertNotNull(fractionField0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = new Fraction(1.0);
      assertEquals((short)1, fraction0.shortValue());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = new Fraction((-790.48563233));
      long long0 = fraction0.longValue();
      assertEquals((-790.48566F), fraction0.floatValue(), 0.01F);
      assertEquals(348, fraction0.getDenominator());
      assertEquals((-790L), long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = fraction0.ZERO.divide(fraction0);
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = new Fraction(3, 3);
      Fraction fraction1 = fraction0.subtract(3);
      assertEquals((-2), fraction1.getNumerator());
      assertEquals(1, fraction0.getDenominator());
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = new Fraction((-790.48563233));
      int int0 = fraction0.intValue();
      assertEquals((-790.48566F), fraction0.floatValue(), 0.01F);
      assertEquals(348, fraction0.getDenominator());
      assertEquals((-790), int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      fraction0.hashCode();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      int int0 = fraction0.getNumerator();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      float float0 = fraction0.floatValue();
      assertEquals(0.25F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = new Fraction(817.059, Integer.MAX_VALUE);
      assertEquals(817.059, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(2.85040095144011776E17, 288);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 285,040,095,144,011,776 to fraction (285,040,095,144,011,776/1)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) Integer.MAX_VALUE, Integer.MAX_VALUE);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert 2,147,483,647 to fraction (9,223,372,034,707,292,162/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-2147483646), (-2147483646), (-2147483646));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Overflow trying to convert -2,147,483,646 to fraction (2,147,483,647/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0.4839999999999236, 0.4839999999999236, (-1241));
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: Unable to convert 0.484 to fraction after -1,241 iterations
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Fraction fraction0 = new Fraction((-2736.235), (-2616));
      String string0 = fraction0.toString();
      assertEquals("-2737", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
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
  public void test23()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(132, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 132/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
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
  public void test25()  throws Throwable  {
      Fraction fraction0 = new Fraction((-220), (-220));
      assertEquals(1.0F, fraction0.floatValue(), 0.01F);
      assertEquals(1, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.MINUS_ONE.abs();
      assertEquals(1, fraction1.getDenominator());
      assertEquals(1.0, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      Fraction fraction1 = Fraction.ONE_FIFTH;
      int int0 = fraction1.compareTo(fraction0);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      Fraction fraction1 = Fraction.ONE_FIFTH;
      int int0 = fraction0.compareTo(fraction1);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = new Fraction(75, 75);
      Fraction fraction1 = Fraction.ONE;
      boolean boolean0 = fraction0.equals(fraction1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      boolean boolean0 = fraction0.equals(fraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      boolean boolean0 = fraction0.equals("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_THIRDS;
      Fraction fraction1 = Fraction.ZERO;
      boolean boolean0 = fraction0.equals(fraction1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = Fraction.ONE_THIRD;
      boolean boolean0 = fraction1.equals(fraction0);
      assertFalse(fraction0.equals((Object)fraction1));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, 37);
      // Undeclared exception!
      try { 
        fraction0.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/37, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_THIRD;
      // Undeclared exception!
      try { 
        fraction0.subtract((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertTrue(fraction1.equals((Object)fraction0));
      assertNotSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = fraction0.ZERO.add(fraction0);
      assertSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO_THIRDS;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.subtract(fraction1);
      assertSame(fraction2, fraction0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = Fraction.ONE_THIRD;
      Fraction fraction2 = fraction0.ONE_THIRD.add(fraction1);
      assertEquals(0.6666667F, fraction2.floatValue(), 0.01F);
      assertEquals(2, fraction2.getNumerator());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertEquals(1, fraction1.getDenominator());
      assertEquals(0.0, fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MAX_VALUE, (-1012));
      Fraction fraction1 = Fraction.ONE_HALF;
      // Undeclared exception!
      try { 
        fraction0.subtract(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: -2,147,484,153
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      // Undeclared exception!
      try { 
        fraction0.ZERO.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Fraction fraction0 = Fraction.THREE_FIFTHS;
      Fraction fraction1 = fraction0.divide(fraction0);
      assertEquals(1, fraction1.getNumerator());
      assertEquals((short)1, fraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.TWO.multiply(fraction1);
      assertEquals(0, fraction2.getNumerator());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_FIFTH;
      // Undeclared exception!
      try { 
        fraction0.ZERO.divide((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // fraction
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.ZERO.divide(fraction0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
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
  public void test49()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, (-3797));
      assertEquals(0, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
      assertEquals(1, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(1, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction 1/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-19681));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-19,681, cannot negate
         //
         verifyException("org.apache.commons.math3.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE_QUARTER;
      String string0 = fraction0.toString();
      assertEquals("1 / 4", string0);
  }
}
