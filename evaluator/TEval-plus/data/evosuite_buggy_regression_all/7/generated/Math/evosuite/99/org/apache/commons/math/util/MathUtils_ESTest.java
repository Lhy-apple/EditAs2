/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 20:04:07 GMT 2023
 */

package org.apache.commons.math.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.util.MathUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MathUtils_ESTest extends MathUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      int int0 = MathUtils.hash(4.9E-324);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      int int0 = MathUtils.hash((double[]) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      float float0 = MathUtils.round((float) (-1073741824), (-1073741824), 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck(9223372036854774991L, 20922789888000L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      double double0 = MathUtils.sinh((-49.25393));
      assertEquals((-1.2293626527160572E21), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      double double0 = MathUtils.normalizeAngle((-1790.2457482076793), (-1790.2457482076793));
      assertEquals((-1790.2457482076793), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 5);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      double double0 = MathUtils.cosh(21);
      assertEquals(6.594078672416073E8, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      double double0 = MathUtils.log(1.0, 0);
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck((-1073741854), (-1073741854));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.addAndCheck(2147483642, 2147483642);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: add
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      int int0 = MathUtils.addAndCheck(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      long long0 = MathUtils.subAndCheck(1778L, 3205L);
      assertEquals((-1427L), long0);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((-30L), 1032L);
      assertEquals((-1062L), long0);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck((-9223372036854775001L), (long) 1896);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((long) 0, (long) 0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(52, 0);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient((byte)1, 14);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient((-1222), (-1222));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(17, 17);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficient(2356, 1489);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(2373, 1);
      assertEquals(2373L, long0);
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(0, (-1));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1, (-1315));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(19, 10);
      assertEquals(92378L, long0);
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(66, 3);
      assertEquals(45760.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      long long0 = MathUtils.binomialCoefficient(71, (-1327));
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientDouble(4, 364);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientDouble((-1136), (-2146989856));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(1952257861, 1952257861);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(4, 0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(35, 1);
      assertEquals(35.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(0, (-1));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientDouble(1896, 1650);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientLog((-2023), 431);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= k for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.binomialCoefficientLog((-257), (-257));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for binomial coefficient (n,k)
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(66, 66);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1896, 0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(3902, 1);
      assertEquals(8.269244521183056, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(0, (-1));
      assertEquals(Double.NEGATIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(80, (-2971));
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      double double0 = MathUtils.binomialCoefficientLog(1896, 1703);
      assertEquals(620.2971441037788, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) Float.NaN, (-2.156266984069E7));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[5] = (double) Float.NaN;
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((-902.090180183854), (-902.090180183854), (-902.090180183854));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((-2003.678216), (-1.0), (double) (-1));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double) 1, (-49.25393), (double) 1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      boolean boolean0 = MathUtils.equals(1511.2, (double) 6400, (double) 6400);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      boolean boolean0 = MathUtils.equals(493.1618596857, 1.4350163065311268, 493.1618596857);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      boolean boolean0 = MathUtils.equals((double[]) null, (double[]) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      boolean boolean0 = MathUtils.equals(doubleArray0, (double[]) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      boolean boolean0 = MathUtils.equals((double[]) null, doubleArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[] doubleArray1 = new double[7];
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (double) (-2963L);
      double[] doubleArray1 = new double[2];
      boolean boolean0 = MathUtils.equals(doubleArray0, doubleArray1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial((-34));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorial(21);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // factorial value is too large to fit in a long
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      double double0 = MathUtils.factorialDouble(0);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialDouble((-15));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n >= 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      double double0 = MathUtils.factorialDouble((byte)56);
      assertEquals(7.109985878048745E74, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.factorialLog((-2640));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // must have n > 0 for n!
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      double double0 = MathUtils.factorialLog(18);
      assertEquals(36.39544520803305, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      int int0 = MathUtils.gcd(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      int int0 = MathUtils.gcd(1548, 0);
      assertEquals(1548, int0);
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.gcd(Integer.MIN_VALUE, Integer.MIN_VALUE);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: gcd(-2,147,483,648, -2,147,483,648) is 2^31
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte) (-1));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      byte byte0 = MathUtils.indicator((byte)92);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      double double0 = MathUtils.indicator((double) 6);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      double double0 = MathUtils.indicator((double) Float.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      double double0 = MathUtils.indicator((double) (-2190.1243F));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      int int0 = MathUtils.indicator((-1194));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      int int0 = MathUtils.indicator(1073741824);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      long long0 = MathUtils.indicator((-923L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      long long0 = MathUtils.indicator(9223372036854774991L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      short short0 = MathUtils.indicator((short) (-1));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      short short0 = MathUtils.indicator((short)788);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      int int0 = MathUtils.lcm(0, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      int int0 = MathUtils.lcm((-1073741824), (-1073741824));
      assertEquals(1073741824, int0);
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      int int0 = MathUtils.lcm(30, 0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.lcm(1305, (-1073741824));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck(2147475138, 17);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: mul
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((long) 798, (-923L));
      assertEquals((-736554L), long0);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-6194339850137525L), (-6194339850137525L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((-246L), (-1L));
      assertEquals(246L, long0);
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck((-1483L), 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.mulAndCheck((-9223372036854775639L), 479001600L);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: multiply
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      long long0 = MathUtils.mulAndCheck(1717988128L, (long) 0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      float float0 = MathUtils.round((float) 2, (-504), 2);
      assertEquals(Float.POSITIVE_INFINITY, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      float float0 = MathUtils.round(5257.4937F, (-1));
      assertEquals(5260.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      float float0 = MathUtils.round((float) (byte)10, (-3245), (int) (byte)1);
      assertEquals(Float.NEGATIVE_INFINITY, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      float float0 = MathUtils.round((float) 5, 5, 5);
      assertEquals(5.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      double double0 = MathUtils.nextAfter(0.9999999999999999, 5.999999908945924E31);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      double double0 = MathUtils.nextAfter(1, (-1073.8980000000001));
      assertEquals(0.9999999999999999, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      double double0 = MathUtils.scalb(0.0, 32);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      double double0 = MathUtils.scalb(Double.NaN, (byte)1);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      double double0 = MathUtils.scalb((-4350.530603182633), 1);
      assertEquals((-8701.061206365266), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      double double0 = MathUtils.scalb(Double.POSITIVE_INFINITY, 1);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      double double0 = MathUtils.round((double) Float.NaN, 0, 1);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      double double0 = MathUtils.round(Double.POSITIVE_INFINITY, 10);
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 1, 0);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      float float0 = MathUtils.round((-1090.4F), 1073741824, 3);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      float float0 = MathUtils.round((float) 6, 6, 7);
      assertEquals(6.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round((float) (-1073741824), 1, 1073741824);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid rounding method.
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      float float0 = MathUtils.round((float) (-1073741850), 1548, 2);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      float float0 = MathUtils.round(Float.NaN, 3078, 3);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      float float0 = MathUtils.round((float) 2749, 2749, 5);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      float float0 = MathUtils.round(2542.4094F, 2, 6);
      assertEquals(2542.41F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      float float0 = MathUtils.round((float) 3078, 3078, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      float float0 = MathUtils.round((float) 0, 2749, 6);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.round(6.0F, (-25), 7);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // Inexact result from rounding
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)77);
      assertEquals((byte)1, byte0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte)0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      byte byte0 = MathUtils.sign((byte) (-44));
      assertEquals((byte) (-1), byte0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      double double0 = MathUtils.sign((double) 1);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      double double0 = MathUtils.sign((double) Float.NaN);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      double double0 = MathUtils.sign(0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      double double0 = MathUtils.sign((-2007.088));
      assertEquals((-1.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      float float0 = MathUtils.sign((-2190.1243F));
      assertEquals((-1.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      float float0 = MathUtils.sign(Float.NaN);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      float float0 = MathUtils.sign(0.0F);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      float float0 = MathUtils.sign((float) 2135212312);
      assertEquals(1.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      int int0 = MathUtils.sign(12100);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      int int0 = MathUtils.sign(0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      int int0 = MathUtils.sign((-1073741833));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test123()  throws Throwable  {
      long long0 = MathUtils.sign((-791L));
      assertEquals((-1L), long0);
  }

  @Test(timeout = 4000)
  public void test124()  throws Throwable  {
      long long0 = MathUtils.sign(0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test125()  throws Throwable  {
      long long0 = MathUtils.sign(438L);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test126()  throws Throwable  {
      short short0 = MathUtils.sign((short)788);
      assertEquals((short)1, short0);
  }

  @Test(timeout = 4000)
  public void test127()  throws Throwable  {
      short short0 = MathUtils.sign((short)0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test128()  throws Throwable  {
      short short0 = MathUtils.sign((short) (-1635));
      assertEquals((short) (-1), short0);
  }

  @Test(timeout = 4000)
  public void test129()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(Integer.MIN_VALUE, 1);
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test130()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck(1073741824, (-1073741824));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test131()  throws Throwable  {
      int int0 = MathUtils.subAndCheck(1896, 1896);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test132()  throws Throwable  {
      long long0 = MathUtils.subAndCheck((-2513865368453184L), (-9223372036854775639L));
      assertEquals(9220858171486322455L, long0);
  }

  @Test(timeout = 4000)
  public void test133()  throws Throwable  {
      // Undeclared exception!
      try { 
        MathUtils.subAndCheck((long) 1418, (-9223372036854775639L));
        fail("Expecting exception: ArithmeticException");
      
      } catch(ArithmeticException e) {
         //
         // overflow: subtract
         //
         verifyException("org.apache.commons.math.util.MathUtils", e);
      }
  }
}