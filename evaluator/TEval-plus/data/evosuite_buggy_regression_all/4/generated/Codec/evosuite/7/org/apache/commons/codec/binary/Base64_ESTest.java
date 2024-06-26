/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:27:40 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import org.apache.commons.codec.binary.Base64;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64_ESTest extends Base64_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("U@");
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("/E}^Y%");
      Base64.decodeInteger(byteArray0);
      assertArrayEquals(new byte[] {(byte) (-4), (byte)70}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Base64 base64_0 = new Base64();
      String string0 = base64_0.encodeToString((byte[]) null);
      assertFalse(base64_0.isUrlSafe());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      String string0 = Base64.encodeBase64String((byte[]) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Parameter Fupplied to B9se64 encode is not a byte[]");
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("ParameterFuppliedtoB9se64encodeisnotabyt", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)81}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      boolean boolean0 = base64_0.hasData();
      assertTrue(base64_0.isUrlSafe());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base64 base64_0 = new Base64(52, (byte[]) null);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[5];
      base64_0.encode(byteArray0, 3588, (-1354));
      boolean boolean0 = base64_0.hasData();
      assertFalse(base64_0.isUrlSafe());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      Base64 base64_0 = new Base64(1512, byteArray0, false);
      int int0 = base64_0.avail();
      assertEquals(0, int0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      Base64 base64_0 = new Base64(112, byteArray0, false);
      int int0 = base64_0.readResults(byteArray0, (byte) (-11), (byte)0);
      assertEquals(0, int0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Base64 base64_0 = new Base64(99);
      byte[] byteArray0 = new byte[4];
      byte[] byteArray1 = base64_0.encode(byteArray0);
      int int0 = base64_0.readResults(byteArray1, 64, 99);
      assertEquals(10, int0);
      assertEquals(10, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)61, (byte)61, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64((byte)5, byteArray0);
      String string0 = base64_0.encodeToString(byteArray0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals("AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      byte[] byteArray0 = base64_0.decode("z4lW");
      int int0 = base64_0.readResults(byteArray0, (-782), (-782));
      assertFalse(base64_0.isUrlSafe());
      assertArrayEquals(new byte[] {(byte) (-49), (byte) (-119), (byte)86}, byteArray0);
      assertEquals(3, byteArray0.length);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer((byte[]) null, 35, 35);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[2];
      base64_0.setInitialBuffer(byteArray0, (-3035), (-3035));
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Base64 base64_0 = new Base64(78);
      byte[] byteArray0 = base64_0.decode("uh~=-y[U8Nb>Y]i/!");
      base64_0.encode(byteArray0, 78, 78);
      assertArrayEquals(new byte[] {(byte) (-70)}, byteArray0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Base64 base64_0 = new Base64(1);
      Object object0 = base64_0.decode((Object) "lineSeperator must not contain base64 characters: [");
      Object object1 = base64_0.encode(object0);
      assertNotSame(object1, object0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AAAAAAA", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      Base64 base64_0 = new Base64(true);
      base64_0.encode(byteArray0, (-1982), (-1982));
      assertEquals(1, byteArray0.length);
      assertTrue(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      // Undeclared exception!
      try { 
        base64_0.encode((byte[]) null, 76, 64);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Parameter Fupplied to B9se64 encode is not a byte[]");
      Base64 base64_0 = new Base64(111);
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, (-1), 111);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("H");
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
      assertEquals(4, byteArray0.length);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Idj|V|e<\"zC)r");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(815, byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [!\uFFFD\uFFFD{0\uFFFD]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Base64 base64_0 = new Base64(99);
      Object object0 = base64_0.decode((Object) "");
      Object object1 = base64_0.decode(object0);
      assertSame(object1, object0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      Base64 base64_0 = new Base64((-795));
      try { 
        base64_0.decode((Object) bigInteger0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64((String) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("_?k80nb:dUp2LQa_");
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, true, false, (-1400));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (18) than the specified maxium size of -1400
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      byte[] byteArray0 = base64_0.decode("Cch{c8!%9A>r~Pzm!e");
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertFalse(base64_0.isUrlSafe());
      assertArrayEquals(new byte[] {(byte) (-56), (byte)92, (byte) (-13), (byte) (-48), (byte)43, (byte)63, (byte)57, (byte) (-98)}, byteArray1);
      assertEquals(8, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte) (-56), (byte)92}, byteArray1);
      assertEquals(2, byteArray1.length);
      assertNotSame(byteArray1, byteArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[5] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(5, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      Base64 base64_0 = new Base64((-1));
      try { 
        base64_0.encode((Object) bigInteger0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Base64 base64_0 = new Base64(99);
      byte[] byteArray0 = Base64.decodeBase64("U@");
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      // Undeclared exception!
      try { 
        Base64.encodeInteger((BigInteger) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // encodeInteger called with null parameter
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      BigInteger bigInteger0 = new BigInteger(byteArray0);
      byte[] byteArray1 = Base64.encodeInteger(bigInteger0);
      assertEquals(0, byteArray1.length);
  }
}
