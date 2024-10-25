/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:21:49 GMT 2023
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
      byte[] byteArray0 = new byte[2];
      BigInteger bigInteger0 = Base64.decodeInteger(byteArray0);
      byte[] byteArray1 = Base64.encodeInteger(bigInteger0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      Base64 base64_0 = new Base64(471, byteArray0);
      Object object0 = base64_0.decode((Object) "UTF-16");
      Object object1 = base64_0.decode(object0);
      assertFalse(base64_0.isUrlSafe());
      assertNotSame(object1, object0);
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("\">P(i");
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("Pg==\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64.encodeBase64URLSafeString(byteArray0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("\">P(i");
      Base64.encodeBase64URLSafe(byteArray0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64.encodeBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      Base64.isArrayByteBase64(byteArray1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("GsV\"uoY-xT");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(111, byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [\u001A\uFFFDn\uFFFD\uFFFD\uFFFD]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.hasData();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("\">P(i");
      Base64 base64_0 = new Base64((byte)84, byteArray0);
      base64_0.encodeToString(byteArray0);
      base64_0.hasData();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base64 base64_0 = new Base64((byte)115);
      base64_0.avail();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64();
      base64_0.encode(byteArray0, (-3618), (-3618));
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64();
      base64_0.readResults(byteArray0, 1341, 1341);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64((byte)5, byteArray0, false);
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64();
      base64_0.decode(byteArray0);
      base64_0.readResults(byteArray0, 1341, 1341);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Base64 base64_0 = new Base64(5, (byte[]) null, false);
      base64_0.setInitialBuffer((byte[]) null, 5, 5);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      Base64 base64_0 = new Base64((byte)1, byteArray0);
      base64_0.setInitialBuffer(byteArray0, (byte)41, 2788);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64(3155);
      base64_0.decode((Object) "AAAA");
      base64_0.encode(byteArray0, (-718), 1825);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.valueOf((-3505L));
      Base64.encodeInteger(bigInteger0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base64.encodeBase64URLSafeString(byteArray0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Base64 base64_0 = new Base64();
      // Undeclared exception!
      try { 
        base64_0.encode((byte[]) null, 13, 52);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("\"P(i");
      Base64 base64_0 = new Base64(1);
      base64_0.encode(byteArray0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Base64.decodeBase64("3':a&H?=+j9_=KQC");
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(16);
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, 16, 16);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 16
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("ISO-8859-1");
      Base64 base64_0 = new Base64(1, byteArray0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Base64 base64_0 = new Base64();
      try { 
        base64_0.decode((Object) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Base64.decodeBase64((String) null);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Base64 base64_0 = new Base64((byte)115);
      base64_0.decode("");
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      String string0 = Base64.encodeBase64URLSafeString((byte[]) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("BR\"*[]4");
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, 0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (6) than the specified maxium size of 0
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.discardWhitespace(byteArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("CeI,M#:4;rhw7");
      Base64.discardWhitespace(byteArray0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertNotSame(byteArray1, byteArray0);
      assertArrayEquals(new byte[] {(byte)62, (byte)103}, byteArray1);
      assertEquals(2, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[6] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(7, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)32, (byte)0}, byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      Base64 base64_0 = new Base64(471, byteArray0);
      Object object0 = base64_0.decode((Object) "UTF-16");
      Object object1 = base64_0.encode(object0);
      assertNotSame(object1, object0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("\">P(i");
      Base64 base64_0 = new Base64((byte)84, byteArray0);
      try { 
        base64_0.encode((Object) "Pg==\r\n");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Base64 base64_0 = new Base64();
      String string0 = base64_0.encodeToString((byte[]) null);
      assertFalse(base64_0.isUrlSafe());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      Base64 base64_0 = new Base64(88, byteArray0);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertSame(byteArray1, byteArray0);
      assertFalse(base64_0.isUrlSafe());
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
}
