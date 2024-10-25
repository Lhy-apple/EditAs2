/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:53 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.compress.archivers.tar.TarUtils;
import org.apache.commons.compress.archivers.zip.ZipEncoding;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarUtils_ESTest extends TarUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((byte)0, byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("\"jee6<H", byteArray0, 102, (int) (byte) (-1));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes((-2750), (byte[]) null, (-2750), (-2750));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -2750=1777777777777777772502 will not fit in octal number buffer of length -2752
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[200];
      TarUtils.formatLongOctalOrBinaryBytes((-166L), byteArray0, (byte)2, (byte)2);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 3, 3);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 90 at offset 0 in 'Z{NUL}{NUL}' len=3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((byte)2, byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2=2 will not fit in octal number buffer of length 0
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)1, (int) (byte)1);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("Vc", byteArray0, (int) (byte)0, (-3629), zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals((-3629), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Length 0 must be at least 2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[32];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((byte)2, byteArray0, (byte)2, (byte)2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 3, (byte)2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((byte)2, byteArray0, (byte)2, (byte)2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, (byte)2, (byte)2);
      assertEquals(2L, long0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)50, (byte)32, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[269];
      TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, (byte)45, (byte)45);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (byte)45, (byte)45);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -128 at offset 0 in '\uFFFD{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}\u0001{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}' len=45
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[269];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, (byte)45, (byte)45);
      assertEquals(90, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)45, (byte)45);
      assertEquals(281474976710656L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[249];
      TarUtils.formatLongOctalOrBinaryBytes((-939L), byteArray0, (byte)67, (byte)67);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)67, 134);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 67, 134 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-7L), byteArray0, (byte)2, (byte)2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)2, (byte)2);
      assertEquals((-7L), long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte) (-125);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-125));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[90];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-3757L), byteArray0, 41, 41);
      assertEquals(82, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 41, 41);
      assertEquals((-3757L), long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[1] = (byte)1;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)1, (int) (byte)1);
      assertEquals("\u0001", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("Vc", byteArray0, (int) (byte)86, 2146235776, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(8589934591L, byteArray0, (-4408), 8);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -4401
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(1099511627776L, byteArray0, (byte)2, (byte)2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1099511627776 is too large for 2 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[249];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2700L), byteArray0, 70, 79);
      assertEquals(149, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[213];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(79, byteArray0, 79, 79);
      assertEquals(158, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte)39;
      byteArray0[1] = (byte)63;
      byteArray0[2] = (byte) (-83);
      byteArray0[3] = (byte) (-19);
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}
