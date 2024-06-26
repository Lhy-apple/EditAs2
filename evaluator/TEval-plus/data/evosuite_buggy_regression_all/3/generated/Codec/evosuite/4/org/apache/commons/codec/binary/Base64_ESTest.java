/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:25:22 GMT 2023
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
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      String string0 = base64_0.encodeToString((byte[]) null);
      assertFalse(base64_0.isUrlSafe());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("AAAAAAAAAAAA\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("");
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64URLSafe((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64((-1), byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [AQ==]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      try { 
        base64_0.decode((Object) base64_0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(true);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertArrayEquals(new byte[] {(byte)81, (byte)86, (byte)69, (byte)57, (byte)80, (byte)81, (byte)13, (byte)10}, byteArray1);
      assertArrayEquals(new byte[] {(byte)65, (byte)81, (byte)61, (byte)61}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64();
      boolean boolean0 = base64_0.hasData();
      assertFalse(boolean0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      assertEquals(0, byteArray1.length);
      
      Base64 base64_0 = new Base64((-901), byteArray1);
      base64_0.encode(byteArray0, (-487), (-901));
      boolean boolean0 = base64_0.hasData();
      assertFalse(base64_0.isUrlSafe());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Base64 base64_0 = new Base64((-613));
      int int0 = base64_0.avail();
      assertFalse(base64_0.isUrlSafe());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64((-685));
      int int0 = base64_0.readResults(byteArray0, (-685), (-685));
      assertFalse(base64_0.isUrlSafe());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64 base64_0 = new Base64();
      byte[] byteArray1 = base64_0.encode(byteArray0);
      int int0 = base64_0.readResults(byteArray1, (byte)101, 65);
      assertEquals(10, int0);
      assertEquals(10, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)61, (byte)13, (byte)10}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64((byte)5);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertEquals(18, byteArray1.length);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64(false);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      int int0 = base64_0.readResults(byteArray1, (byte)1, (-374));
      assertEquals(14, byteArray1.length);
      assertFalse(base64_0.isUrlSafe());
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64((-1172), byteArray0);
      base64_0.setInitialBuffer((byte[]) null, (-1), 1108);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      Base64 base64_0 = new Base64((-3944));
      base64_0.setInitialBuffer(byteArray0, 80, 80);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base64 base64_0 = new Base64((-398));
      Object object0 = base64_0.decode((Object) "|>'X^hg\"gc@^3M[1>");
      base64_0.encode(byteArray0, (-2637), (-1126));
      assertFalse(byteArray0.equals((Object)object0));
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, true, true, 33);
      assertEquals(13, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.encode((byte[]) null, (-2316), (-2316));
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      Base64 base64_0 = new Base64(982, byteArray0);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, 982, 982);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 982
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer(byteArray0, 1, 1);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, (-1071), 96);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1071
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      byte[] byteArray2 = Base64.encodeBase64(byteArray1, true);
      assertEquals(10, byteArray2.length);
      assertArrayEquals(new byte[] {(byte)81, (byte)85, (byte)69, (byte)57, (byte)80, (byte)86, (byte)52, (byte)75, (byte)94, (byte)10}, byteArray2);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Base64 base64_0 = new Base64(16, (byte[]) null);
      // Undeclared exception!
      try { 
        base64_0.decode((byte[]) null, 16, 1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = base64_0.decode("K8`B)=hM~1&4lk");
      assertEquals(2, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)43, (byte) (-64)}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[8];
      byteArray0[2] = (byte) (-71);
      byte[] byteArray1 = base64_0.decode(byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Base64 base64_0 = new Base64((-613));
      Object object0 = base64_0.decode((Object) "UTF-16");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)114;
      byteArray0[1] = (byte)125;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Base64 base64_0 = new Base64((-398));
      Object object0 = base64_0.decode((Object) "|>'X^hg\"gc@^3M[1>");
      Object object1 = base64_0.decode(object0);
      assertNotSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      // Undeclared exception!
      try { 
        Base64.decodeInteger((byte[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, true, (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (14) than the specified maxium size of -1
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[2] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)9, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(8, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      byte[] byteArray2 = Base64.discardWhitespace(byteArray1);
      assertEquals(14, byteArray2.length);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[3] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(3, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Base64 base64_0 = new Base64(3863);
      Object object0 = base64_0.decode((Object) "a");
      Object object1 = base64_0.encode(object0);
      assertSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      Base64 base64_0 = new Base64((-901), byteArray1);
      Object object0 = new Object();
      try { 
        base64_0.encode(object0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertEquals(8, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
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
  public void test42()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      assertEquals(0, byteArray0.length);
  }
}
