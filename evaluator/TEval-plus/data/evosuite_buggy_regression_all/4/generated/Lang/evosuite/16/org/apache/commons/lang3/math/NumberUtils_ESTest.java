/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:01:24 GMT 2023
 */

package org.apache.commons.lang3.math;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.apache.commons.lang3.math.NumberUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumberUtils_ESTest extends NumberUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test000()  throws Throwable  {
      float float0 = NumberUtils.max((-1916.6F), (float) 0L, (float) (-752L));
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test001()  throws Throwable  {
      float float0 = NumberUtils.min((float) (byte) (-46), (float) (byte) (-46), (float) (byte) (-46));
      assertEquals((-46.0F), float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test002()  throws Throwable  {
      double double0 = NumberUtils.toDouble("HK*Y");
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test003()  throws Throwable  {
      double double0 = NumberUtils.min(0.0, 0.0, 0.0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test004()  throws Throwable  {
      NumberUtils numberUtils0 = new NumberUtils();
      byte[] byteArray0 = new byte[8];
      byteArray0[0] = (byte) numberUtils0.BYTE_ONE;
      byte byte0 = NumberUtils.min(byteArray0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test005()  throws Throwable  {
      byte byte0 = NumberUtils.toByte((String) null);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test006()  throws Throwable  {
      double double0 = NumberUtils.max((-708.7255117), (-19.0451953503), (-708.7255117));
      assertEquals((-19.0451953503), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test007()  throws Throwable  {
      long long0 = NumberUtils.toLong("c^KfF[o.sq!8m");
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test008()  throws Throwable  {
      int int0 = NumberUtils.toInt((String) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test009()  throws Throwable  {
      float float0 = NumberUtils.toFloat("k-\",QVt `-jx$e=:G");
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test010()  throws Throwable  {
      short short0 = NumberUtils.toShort((String) null);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test011()  throws Throwable  {
      int int0 = NumberUtils.toInt("fd");
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test012()  throws Throwable  {
      long long0 = NumberUtils.toLong((String) null);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test013()  throws Throwable  {
      float float0 = NumberUtils.toFloat((String) null, (float) 0);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test014()  throws Throwable  {
      double double0 = NumberUtils.toDouble((String) null, (double) 5);
      assertEquals(5.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test015()  throws Throwable  {
      byte byte0 = NumberUtils.toByte(".R|?");
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test016()  throws Throwable  {
      short short0 = NumberUtils.toShort("The Array must not be null", (short)0);
      assertEquals((short)0, short0);
  }

  @Test(timeout = 4000)
  public void test017()  throws Throwable  {
      Number number0 = NumberUtils.createNumber((String) null);
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test018()  throws Throwable  {
      try { 
        NumberUtils.createNumber("");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // A blank string is not a valid number
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test019()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("--3T/:q'y[DL");
      assertNull(number0);
  }

  @Test(timeout = 4000)
  public void test020()  throws Throwable  {
      try { 
        NumberUtils.createNumber("0x-0x");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Sign character in wrong position
         //
         verifyException("java.lang.Integer", e);
      }
  }

  @Test(timeout = 4000)
  public void test021()  throws Throwable  {
      try { 
        NumberUtils.createNumber("-0xk-\",QVt `-jx$e=:G");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"-k-\",QVt `-jx$e=:G\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test022()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("0.");
      assertEquals(0.0F, number0);
  }

  @Test(timeout = 4000)
  public void test023()  throws Throwable  {
      try { 
        NumberUtils.createNumber("0nFxl.k9eUE");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // 0nFxl.k9eUE is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test024()  throws Throwable  {
      try { 
        NumberUtils.createNumber("Array cannot be empty.");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Array cannot be empty. is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test025()  throws Throwable  {
      try { 
        NumberUtils.createNumber("Oa]n.e `Fy)g!-y{[");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // Oa]n.e `Fy)g!-y{[ is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test026()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("0F");
      assertEquals((byte)0, number0.byteValue());
  }

  @Test(timeout = 4000)
  public void test027()  throws Throwable  {
      try { 
        NumberUtils.createNumber("CvNs'?IXce)9wUIjkE");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // CvNs'?IXce)9wUIjkE is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test028()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("0e2");
      assertEquals((byte)0, number0.byteValue());
  }

  @Test(timeout = 4000)
  public void test029()  throws Throwable  {
      try { 
        NumberUtils.createNumber("0e");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // 0e is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test030()  throws Throwable  {
      try { 
        NumberUtils.createNumber("0eD");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // 0eD is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test031()  throws Throwable  {
      try { 
        NumberUtils.createNumber("-GL");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // -GL is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test032()  throws Throwable  {
      try { 
        NumberUtils.createNumber("d");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // d is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test033()  throws Throwable  {
      try { 
        NumberUtils.createNumber("'yupT1yS>1f");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // 'yupT1yS>1f is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test034()  throws Throwable  {
      try { 
        NumberUtils.createNumber("N 6/r(UOkMfRu.dql");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // N 6/r(UOkMfRu.dql is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test035()  throws Throwable  {
      try { 
        NumberUtils.createNumber("The Array must not be null");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // The Array must not be null is not a valid number.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test036()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("4l");
      assertEquals(4L, number0);
  }

  @Test(timeout = 4000)
  public void test037()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("-0l");
      assertEquals(0L, number0);
  }

  @Test(timeout = 4000)
  public void test038()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("2F");
      assertEquals(2.0F, number0);
  }

  @Test(timeout = 4000)
  public void test039()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("70D");
      assertEquals(70.0, number0);
  }

  @Test(timeout = 4000)
  public void test040()  throws Throwable  {
      try { 
        NumberUtils.createNumber("E6e5");
        fail("Expecting exception: NumberFormatException");
      
      } catch(NumberFormatException e) {
         //
         // For input string: \"E6e5\"
         //
         verifyException("java.lang.NumberFormatException", e);
      }
  }

  @Test(timeout = 4000)
  public void test041()  throws Throwable  {
      Number number0 = NumberUtils.createNumber("6e5");
      assertEquals(600000.0F, number0);
  }

  @Test(timeout = 4000)
  public void test042()  throws Throwable  {
      Float float0 = NumberUtils.createFloat((String) null);
      assertNull(float0);
  }

  @Test(timeout = 4000)
  public void test043()  throws Throwable  {
      Double double0 = NumberUtils.createDouble((String) null);
      assertNull(double0);
  }

  @Test(timeout = 4000)
  public void test044()  throws Throwable  {
      Integer integer0 = NumberUtils.createInteger((String) null);
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test045()  throws Throwable  {
      Long long0 = NumberUtils.createLong((String) null);
      assertNull(long0);
  }

  @Test(timeout = 4000)
  public void test046()  throws Throwable  {
      BigInteger bigInteger0 = NumberUtils.createBigInteger((String) null);
      assertNull(bigInteger0);
  }

  @Test(timeout = 4000)
  public void test047()  throws Throwable  {
      BigDecimal bigDecimal0 = NumberUtils.createBigDecimal((String) null);
      assertNull(bigDecimal0);
  }

  @Test(timeout = 4000)
  public void test048()  throws Throwable  {
      long[] longArray0 = new long[2];
      long long0 = NumberUtils.min(longArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test049()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((long[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test050()  throws Throwable  {
      long[] longArray0 = new long[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test051()  throws Throwable  {
      long[] longArray0 = new long[2];
      longArray0[1] = (-2520L);
      long long0 = NumberUtils.min(longArray0);
      assertEquals((-2520L), long0);
  }

  @Test(timeout = 4000)
  public void test052()  throws Throwable  {
      int[] intArray0 = new int[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test053()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((int[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test054()  throws Throwable  {
      int[] intArray0 = new int[4];
      intArray0[0] = 1403;
      int int0 = NumberUtils.min(intArray0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test055()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((short[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test056()  throws Throwable  {
      short[] shortArray0 = new short[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(shortArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test057()  throws Throwable  {
      short[] shortArray0 = new short[4];
      shortArray0[1] = (short) (-311);
      short short0 = NumberUtils.min(shortArray0);
      assertEquals((short) (-311), short0);
  }

  @Test(timeout = 4000)
  public void test058()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byte byte0 = NumberUtils.min(byteArray0);
      assertEquals((byte)0, byte0);
  }

  @Test(timeout = 4000)
  public void test059()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((byte[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test060()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test061()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((double[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test062()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test063()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      doubleArray0[3] = Double.NaN;
      double double0 = NumberUtils.min(doubleArray0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test064()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[2] = (double) (-3046L);
      double double0 = NumberUtils.min(doubleArray0);
      assertEquals((-3046.0), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test065()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.min((float[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test066()  throws Throwable  {
      float[] floatArray0 = new float[0];
      // Undeclared exception!
      try { 
        NumberUtils.min(floatArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test067()  throws Throwable  {
      float[] floatArray0 = new float[7];
      floatArray0[2] = Float.NaN;
      float float0 = NumberUtils.min(floatArray0);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test068()  throws Throwable  {
      float[] floatArray0 = new float[7];
      floatArray0[0] = 1467.7944F;
      float float0 = NumberUtils.min(floatArray0);
      assertEquals(0.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test069()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((long[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test070()  throws Throwable  {
      long[] longArray0 = new long[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(longArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test071()  throws Throwable  {
      NumberUtils numberUtils0 = new NumberUtils();
      long[] longArray0 = new long[9];
      longArray0[0] = (long) (short)numberUtils0.SHORT_MINUS_ONE;
      long long0 = NumberUtils.max(longArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test072()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((int[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test073()  throws Throwable  {
      int[] intArray0 = new int[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(intArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test074()  throws Throwable  {
      int[] intArray0 = new int[4];
      intArray0[1] = 1;
      int int0 = NumberUtils.max(intArray0);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test075()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((short[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test076()  throws Throwable  {
      short[] shortArray0 = new short[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(shortArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test077()  throws Throwable  {
      short[] shortArray0 = new short[4];
      shortArray0[1] = (short)120;
      short short0 = NumberUtils.max(shortArray0);
      assertEquals((short)120, short0);
  }

  @Test(timeout = 4000)
  public void test078()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test079()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((byte[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test080()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[1] = (byte)67;
      byte byte0 = NumberUtils.max(byteArray0);
      assertEquals((byte)67, byte0);
  }

  @Test(timeout = 4000)
  public void test081()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((double[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test082()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test083()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[5] = Double.NaN;
      double double0 = NumberUtils.max(doubleArray0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test084()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      doubleArray0[0] = (-0.8413469576017647);
      double double0 = NumberUtils.max(doubleArray0);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test085()  throws Throwable  {
      // Undeclared exception!
      try { 
        NumberUtils.max((float[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Array must not be null
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test086()  throws Throwable  {
      float[] floatArray0 = new float[0];
      // Undeclared exception!
      try { 
        NumberUtils.max(floatArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array cannot be empty.
         //
         verifyException("org.apache.commons.lang3.math.NumberUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test087()  throws Throwable  {
      float[] floatArray0 = new float[7];
      floatArray0[2] = Float.NaN;
      float float0 = NumberUtils.max(floatArray0);
      assertEquals(Float.NaN, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test088()  throws Throwable  {
      float[] floatArray0 = new float[5];
      floatArray0[1] = 4164.0F;
      float float0 = NumberUtils.max(floatArray0);
      assertEquals(4164.0F, float0, 0.01F);
  }

  @Test(timeout = 4000)
  public void test089()  throws Throwable  {
      long long0 = NumberUtils.min(0L, 0L, 0L);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test090()  throws Throwable  {
      long long0 = NumberUtils.min((-952L), (-2718L), 0L);
      assertEquals((-2718L), long0);
  }

  @Test(timeout = 4000)
  public void test091()  throws Throwable  {
      long long0 = NumberUtils.min((long) (byte) (-83), (-1284L), (long) (-234736464));
      assertEquals((-234736464L), long0);
  }

  @Test(timeout = 4000)
  public void test092()  throws Throwable  {
      int int0 = NumberUtils.min(54, (-374), 54);
      assertEquals((-374), int0);
  }

  @Test(timeout = 4000)
  public void test093()  throws Throwable  {
      int int0 = NumberUtils.min(1431655765, 1431655765, (-234736464));
      assertEquals((-234736464), int0);
  }

  @Test(timeout = 4000)
  public void test094()  throws Throwable  {
      short short0 = NumberUtils.min((short)1327, (short)1327, (short)45);
      assertEquals((short)45, short0);
  }

  @Test(timeout = 4000)
  public void test095()  throws Throwable  {
      short short0 = NumberUtils.min((short)2031, (short) (-3729), (short) (byte)69);
      assertEquals((short) (-3729), short0);
  }

  @Test(timeout = 4000)
  public void test096()  throws Throwable  {
      byte byte0 = NumberUtils.min((byte)108, (byte)102, (byte)108);
      assertEquals((byte)102, byte0);
  }

  @Test(timeout = 4000)
  public void test097()  throws Throwable  {
      byte byte0 = NumberUtils.min((byte)0, (byte)0, (byte) (-15));
      assertEquals((byte) (-15), byte0);
  }

  @Test(timeout = 4000)
  public void test098()  throws Throwable  {
      long long0 = NumberUtils.max(0L, (-3350L), (long) (byte)1);
      assertEquals(1L, long0);
  }

  @Test(timeout = 4000)
  public void test099()  throws Throwable  {
      long long0 = NumberUtils.max((-752L), 2474L, 0L);
      assertEquals(2474L, long0);
  }

  @Test(timeout = 4000)
  public void test100()  throws Throwable  {
      int int0 = NumberUtils.max((-422), (-804), (int) (byte)67);
      assertEquals(67, int0);
  }

  @Test(timeout = 4000)
  public void test101()  throws Throwable  {
      int int0 = NumberUtils.max((-3994), (int) (byte) (-46), (-2416));
      assertEquals((-46), int0);
  }

  @Test(timeout = 4000)
  public void test102()  throws Throwable  {
      short short0 = NumberUtils.max((short) (-594), (short) (-594), (short)4);
      assertEquals((short)4, short0);
  }

  @Test(timeout = 4000)
  public void test103()  throws Throwable  {
      short short0 = NumberUtils.max((short) (-4080), (short)2162, (short) (-4080));
      assertEquals((short)2162, short0);
  }

  @Test(timeout = 4000)
  public void test104()  throws Throwable  {
      byte byte0 = NumberUtils.max((byte)22, (byte)22, (byte)47);
      assertEquals((byte)47, byte0);
  }

  @Test(timeout = 4000)
  public void test105()  throws Throwable  {
      byte byte0 = NumberUtils.max((byte)1, (byte)16, (byte)1);
      assertEquals((byte)16, byte0);
  }

  @Test(timeout = 4000)
  public void test106()  throws Throwable  {
      boolean boolean0 = NumberUtils.isDigits((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test107()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("9Ee2");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test108()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber((String) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test109()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("--");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test110()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0El");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test111()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("-0x6");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test112()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("-0x");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test113()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("-0x&$^)uOb~");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test114()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("-0xbI'06{B?");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test115()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0x68u1$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test116()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0xArray cannot be empty.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test117()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("-");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test118()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("1e+.oKuPVOzQ");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test119()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("...");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test120()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("ez<=*O/k");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test121()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("cjh");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test122()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("----M");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test123()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("6e5");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test124()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("6e");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test125()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("E");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test126()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0e.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test127()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("..");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test128()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber(".");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test129()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("d");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test130()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test131()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("f");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test132()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber(".F");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test133()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("L");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test134()  throws Throwable  {
      boolean boolean0 = NumberUtils.isNumber("0l");
      assertTrue(boolean0);
  }
}