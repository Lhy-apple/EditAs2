/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:32:40 GMT 2023
 */

package org.apache.commons.math.fraction;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.fraction.Fraction;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Fraction_ESTest extends Fraction_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      long long0 = fraction0.longValue();
      assertEquals(2L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.ONE.add(fraction0);
      assertEquals((short)1, fraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      fraction0.hashCode();
      assertEquals((-1), fraction0.getNumerator());
      assertEquals(1, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertTrue(fraction1.equals((Object)fraction0));
      assertNotSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = fraction0.TWO.reciprocal();
      Fraction fraction2 = new Fraction(2970, (-304));
      Fraction fraction3 = fraction1.add(fraction2);
      assertEquals(152, fraction2.getDenominator());
      assertEquals((-1409), fraction3.getNumerator());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Fraction fraction0 = new Fraction(0.0);
      float float0 = fraction0.floatValue();
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) 1, 1);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // Overflow trying to convert 1 to fraction (-9,223,372,036,854,775,808/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      int int0 = fraction0.intValue();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Double.POSITIVE_INFINITY, 1950);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // Overflow trying to convert \u221E to fraction (9,223,372,036,854,775,807/1)
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((double) 23, 23);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // Overflow trying to convert 23 to fraction (9,223,372,036,854,775,786/9,223,372,036,854,775,807)
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Fraction fraction0 = new Fraction((-504.0919040353));
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertEquals((-137113), fraction0.getNumerator());
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Double.NEGATIVE_INFINITY);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // Unable to convert -\u221E to fraction after 100 iterations
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Fraction fraction0 = new Fraction(2.0460629400531616E-5, 4109);
      assertEquals(0, fraction0.getNumerator());
      assertEquals(1, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Fraction fraction0 = new Fraction(2970, (-304));
      assertEquals((-9.769737F), fraction0.floatValue(), 0.01F);
      assertEquals(152, fraction0.getDenominator());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Fraction fraction0 = null;
      try {
        fraction0 = new Fraction((-649), Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -649/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.MINUS_ONE.abs();
      assertEquals(1, fraction1.getNumerator());
      assertEquals((short)1, fraction1.shortValue());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.TWO.reciprocal();
      Fraction fraction2 = fraction1.abs();
      assertEquals(1, fraction1.getNumerator());
      assertEquals(0.5F, fraction2.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      int int0 = fraction0.compareTo(fraction0);
      assertEquals(0, int0);
      assertEquals(1L, fraction0.longValue());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.TWO.reciprocal();
      int int0 = fraction0.compareTo(fraction1);
      assertEquals(1, fraction1.getNumerator());
      assertEquals(0.5, fraction1.doubleValue(), 0.01);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      Fraction fraction1 = Fraction.ZERO;
      int int0 = fraction0.compareTo(fraction1);
      assertEquals(1, int0);
      assertEquals(2.0, fraction0.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      Fraction fraction1 = new Fraction(7.423260171890433E-6, 7.423260171890433E-6, 4800);
      boolean boolean0 = fraction0.equals(fraction1);
      assertFalse(boolean0);
      assertEquals(7.423299E-6F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      boolean boolean0 = fraction0.equals(fraction0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      boolean boolean0 = fraction0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-353), (-353));
      Fraction fraction1 = new Fraction(7.423260171890433E-6, 7.423260171890433E-6, 4800);
      boolean boolean0 = fraction0.equals(fraction1);
      assertEquals(1.0F, fraction0.floatValue(), 0.01F);
      assertFalse(boolean0);
      assertFalse(fraction1.equals((Object)fraction0));
      assertEquals(7.423299E-6F, fraction1.floatValue(), 0.01F);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-10), (-10));
      Fraction fraction1 = Fraction.getReducedFraction(31, 31);
      boolean boolean0 = fraction1.equals(fraction0);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
      assertTrue(boolean0);
      assertEquals(1, fraction1.getNumerator());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Fraction fraction0 = new Fraction(Integer.MIN_VALUE);
      // Undeclared exception!
      try { 
        fraction0.negate();
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/1, cannot negate
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      // Undeclared exception!
      try { 
        fraction0.subtract((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The fraction must not be null
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = fraction0.ZERO.add(fraction0);
      assertSame(fraction1, fraction0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Fraction fraction0 = Fraction.ZERO;
      Fraction fraction1 = Fraction.TWO;
      Fraction fraction2 = fraction0.ONE.add(fraction1);
      assertEquals(3, fraction2.getNumerator());
      assertEquals(3L, fraction2.longValue());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Fraction fraction0 = Fraction.TWO;
      Fraction fraction1 = fraction0.subtract(fraction0);
      assertEquals(0.0F, fraction1.floatValue(), 0.01F);
      assertEquals(1, fraction1.getDenominator());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction((-2147483640), 629);
      Fraction fraction1 = fraction0.negate();
      // Undeclared exception!
      try { 
        fraction0.subtract(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow, numerator too large after multiply: -4,294,967,280
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      // Undeclared exception!
      try { 
        fraction0.ONE.multiply((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The fraction must not be null
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.ZERO.multiply(fraction0);
      assertEquals(0L, fraction1.longValue());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = Fraction.ZERO;
      Fraction fraction2 = fraction0.MINUS_ONE.multiply(fraction1);
      assertEquals(0, fraction2.intValue());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Fraction fraction0 = Fraction.ONE;
      Fraction fraction1 = fraction0.MINUS_ONE.divide(fraction0);
      assertEquals((-1), fraction1.getNumerator());
      assertEquals((-1.0), fraction1.doubleValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      // Undeclared exception!
      try { 
        fraction0.ONE.divide((Fraction) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The fraction must not be null
         //
         verifyException("org.apache.commons.math.fraction.Fraction", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Fraction fraction0 = Fraction.MINUS_ONE;
      Fraction fraction1 = Fraction.ZERO;
      // Undeclared exception!
      try { 
        fraction0.TWO.divide(fraction1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // the fraction to divide by must not be zero: 0/1
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(0, 0);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // zero denominator in fraction 0/0
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(0, 1);
      assertEquals(0, fraction0.intValue());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Fraction fraction0 = Fraction.getReducedFraction(Integer.MIN_VALUE, Integer.MIN_VALUE);
      assertEquals(1.0, fraction0.doubleValue(), 0.01);
      assertEquals(1, fraction0.getNumerator());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction((-379), Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -379/-2,147,483,648, cannot negate
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      // Undeclared exception!
      try { 
        Fraction.getReducedFraction(Integer.MIN_VALUE, (-265));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow in fraction -2,147,483,648/-265, cannot negate
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }
}
