/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:48:22 GMT 2023
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
      Base64 base64_0 = new Base64(1);
      Object object0 = base64_0.decode((Object) "]");
      Object object1 = base64_0.decode(object0);
      assertFalse(base64_0.isUrlSafe());
      assertSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      BigInteger bigInteger0 = Base64.decodeInteger(byteArray0);
      byte[] byteArray1 = Base64.encodeInteger(bigInteger0);
      assertEquals(0, byteArray1.length);
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
      byte[] byteArray0 = new byte[7];
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("AAAAAAAAAA==\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("D");
      Base64 base64_0 = new Base64();
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AAA", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64URLSafe((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      assertEquals(14, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base64 base64_0 = new Base64((-41), (byte[]) null);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("b;N7}l$b&");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64((-2022), byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [l\uFFFD\uFFFD]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base64 base64_0 = new Base64((-800));
      base64_0.hasData();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      Base64 base64_0 = new Base64((-871), byteArray1, false);
      base64_0.encode(byteArray1, 106, (-442));
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Base64 base64_0 = new Base64((-800));
      base64_0.avail();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Base64 base64_0 = new Base64((-1));
      byte[] byteArray0 = new byte[2];
      base64_0.readResults(byteArray0, (byte)35, (byte)35);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64();
      byte[] byteArray1 = base64_0.encode(byteArray0);
      base64_0.readResults(byteArray1, (byte)33, (byte)33);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, true);
      Base64 base64_0 = new Base64((byte)8);
      base64_0.encode(byteArray1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base64 base64_0 = new Base64((-1));
      byte[] byteArray0 = new byte[2];
      base64_0.decode(byteArray0, 12, (-1));
      base64_0.readResults(byteArray0, (byte)35, (byte)35);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer((byte[]) null, 76, 76);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer(byteArray0, (-3553), (byte)0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64((byte)8);
      base64_0.encode(byteArray0);
      base64_0.encode(byteArray0, (-1480), 3065);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base64.encodeBase64URLSafeString(byteArray0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.valueOf((-885L));
      Base64.encodeInteger(bigInteger0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("");
      Base64 base64_0 = new Base64(true);
      base64_0.encode(byteArray0, 2068, (-1039));
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, 2302, 1920);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 2302
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base64 base64_0 = new Base64(1);
      base64_0.encode(byteArray0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Base64 base64_0 = new Base64((byte)39);
      Object object0 = base64_0.decode((Object) "AAAAAAAAAA==\r\n");
      base64_0.encode(object0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("lineSeperator must not cotain base64 characters [");
      Base64 base64_0 = new Base64(9);
      base64_0.decode(byteArray0, 9, 9);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Base64.decodeBase64("!N7T");
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      Base64.isArrayByteBase64(byteArray1);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      boolean boolean0 = Base64.isBase64((byte) (-44));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("UTF-8");
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, true);
      Base64.isArrayByteBase64(byteArray1);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      Base64 base64_0 = new Base64();
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
  public void test33()  throws Throwable  {
      Base64.decodeBase64((String) null);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, (int) (byte) (-25));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (6) than the specified maxium size of -25
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[1] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0}, byteArray1);
      assertEquals(1, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(1, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[2] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(7, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)32, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)111;
      byteArray0[1] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      Base64 base64_0 = new Base64();
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
  public void test41()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AAAA", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
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
}