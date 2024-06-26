/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:17:42 GMT 2023
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
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base64 base64_0 = new Base64(5);
      byte[] byteArray0 = new byte[3];
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("AAAA\r\n", string0);
      
      base64_0.decode(byteArray0, 5, (int) (byte)66);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      BigInteger bigInteger0 = Base64.decodeInteger(byteArray0);
      assertEquals((short)0, bigInteger0.shortValue());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("_&IsYit^");
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("/IsYig==\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AA", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      byte[] byteArray2 = Base64.discardWhitespace(byteArray1);
      assertEquals(6, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)61}, byteArray2);
      assertEquals(4, byteArray2.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base64 base64_0 = new Base64(5099, (byte[]) null);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Input array too big, the output array would be bigger (");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(0, byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [\"zn\uFFFD\uFFFD\uFFFDk+h\uFFFD\uFFFD\uFFFD\uFFFD\u0017\uFFFD\uFFFD\uFFFDn\uFFFD\uFFFD\uFFFDk,(\uFFFDW[y\uFFFD\uFFFD\uFFFD\uFFFD]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      boolean boolean0 = base64_0.hasData();
      assertFalse(base64_0.isUrlSafe());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base64 base64_0 = new Base64(3050);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertEquals(6, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)61, (byte)13, (byte)10}, byteArray1);
      
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      int int0 = base64_0.avail();
      assertEquals(0, int0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64();
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, 1984, 1984);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1984
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64(";EN%}`R`-MD85[3R#U,");
      Base64 base64_0 = new Base64();
      int int0 = base64_0.readResults(byteArray0, 1085, (-425));
      assertArrayEquals(new byte[] {(byte)16, (byte) (-44), (byte)126, (byte)48, (byte)63, (byte)57, (byte) (-35), (byte)21}, byteArray0);
      assertEquals(0, int0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(8, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base64 base64_0 = new Base64();
      byte[] byteArray1 = base64_0.encode(byteArray0);
      int int0 = base64_0.readResults(byteArray1, 3050, 2760);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)61}, byteArray1);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(4, int0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64(";EN%}`R`-MD85[3R#U,");
      Base64 base64_0 = new Base64();
      base64_0.decode((Object) ";EN%}`R`-MD85[3R#U,");
      int int0 = base64_0.readResults(byteArray0, 1085, (-425));
      assertArrayEquals(new byte[] {(byte)16, (byte) (-44), (byte)126, (byte)48, (byte)63, (byte)57, (byte) (-35), (byte)21}, byteArray0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(8, byteArray0.length);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer((byte[]) null, 8, 1097);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      byte[] byteArray0 = new byte[0];
      base64_0.setInitialBuffer(byteArray0, (-1008), (-1008));
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base64 base64_0 = new Base64();
      byte[] byteArray1 = base64_0.encode(byteArray0);
      base64_0.encode(byteArray1, 3050, 3050);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)61}, byteArray1);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(4, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(false);
      base64_0.encode(byteArray0, 1073741824, (-54));
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base64 base64_0 = new Base64((-1047));
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, (-1047), 2508);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1047
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = bigInteger0.toByteArray();
      byte[] byteArray1 = Base64.decodeBase64("}`f");
      Base64 base64_0 = new Base64((byte)0, byteArray0, false);
      base64_0.setInitialBuffer(byteArray1, (byte)0, (byte)0);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray1, (int) (byte)0, 666);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = base64_0.decode((Object) "UTF-16BE");
      Object object1 = base64_0.encode(object0);
      assertNotSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("<=N WrQz`qJ/[D'4j$m");
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = base64_0.decode((Object) "UTF-16BE");
      Object object1 = base64_0.decode(object0);
      assertNotSame(object1, object0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("CV[DWMq:p<r5");
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertArrayEquals(new byte[] {(byte)9, (byte)80, (byte) (-42), (byte)50, (byte) (-86), (byte)107}, byteArray0);
      assertEquals(6, byteArray0.length);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("e8,_5t1G%5-SY~!");
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertEquals(8, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)123, (byte) (-49), (byte) (-7), (byte) (-73), (byte)81, (byte) (-71), (byte) (-7), (byte)38}, byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
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
  public void test29()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64((String) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.decode((Object) "");
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      String string0 = Base64.encodeBase64String((byte[]) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64(";EN%}`R`-MD8*[3R#U,");
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, 1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (14) than the specified maxium size of 1
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[1] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0}, byteArray1);
      assertEquals(1, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[7] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)32, (byte)0}, byteArray0);
      assertEquals(8, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("IO-885-1");
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertArrayEquals(new byte[] {(byte)32, (byte) (-17), (byte) (-68), (byte) (-13), (byte) (-97), (byte) (-75)}, byteArray0);
      assertEquals(6, byteArray0.length);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Base64 base64_0 = new Base64(27);
      try { 
        base64_0.encode((Object) ") than the specified maxium size of ");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Base64 base64_0 = new Base64();
      String string0 = base64_0.encodeToString((byte[]) null);
      assertFalse(base64_0.isUrlSafe());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("_");
      Base64 base64_0 = new Base64((-1661), byteArray0);
      Object object0 = base64_0.decode((Object) "_");
      Object object1 = base64_0.encode(object0);
      assertEquals(0, byteArray0.length);
      assertSame(object0, object1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      String string0 = Base64.encodeBase64URLSafeString(byteArray1);
      assertEquals("QUFBPQAA", string0);
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
