/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:25:21 GMT 2023
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
      byte[] byteArray0 = new byte[168];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8589934591L, byteArray0, 121, 32);
      assertEquals(153, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 121, 32);
      assertEquals(8589934591L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("!%vV.}sEBAaQX", byteArray0, 1265, 1265);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(2, byteArray0, 2, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2=2 will not fit in octal number buffer of length 0
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[153];
      TarUtils.formatLongOctalOrBinaryBytes((-2549L), byteArray0, 32, 32);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 32, 64);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -1 at offset 0 in '\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\u000B{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}' len=64
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((-193L), byteArray0, (byte)76, (byte)76);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 149
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.parseName((byte[]) null, 41, 41);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("p1]A", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      TarUtils.formatUnsignedOctalString(2, byteArray0, 2, 2);
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, 2, 2, zipEncoding0);
      assertEquals("02", string0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)48, (byte)50, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[153];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(60, byteArray0, 60, 60);
      assertEquals(120, int0);
      
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, 3, 120, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 42, (-810));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Length -810 must be at least 2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[118];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 2, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      TarUtils.formatLongOctalOrBinaryBytes(5L, byteArray0, 2, 2);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 3, Integer.MAX_VALUE);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -2147483647
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      TarUtils.formatUnsignedOctalString(2, byteArray0, 2, 2);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 2, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 50 at offset 1 in '02' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(5L, byteArray0, 2, 2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 3, 2);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)53, (byte)32, (byte)0, (byte)0}, byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[127];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(4, byteArray0, 4, 4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 4, 8);
      assertEquals(4L, long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[121];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(2305843009213693935L, byteArray0, 24, 24);
      assertEquals(48, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 24, 24);
      assertEquals(2305843009213693935L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[153];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2549L), byteArray0, 32, 32);
      assertEquals(64, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 32, 32);
      assertEquals((-2549L), long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[127];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-15L), byteArray0, 4, 4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 4, 8);
      assertEquals((-64424509440L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte) (-83);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[153];
      TarUtils.formatLongOctalOrBinaryBytes((-2549L), byteArray0, 32, 32);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 32, 64);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 32, 64 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      String string0 = TarUtils.parseName(byteArray0, 2, 2, (ZipEncoding) null);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      int int0 = TarUtils.formatNameBytes("p1]A", byteArray0, 0, (-1324), zipEncoding0);
      assertEquals((-1324), int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      byte[] byteArray0 = new byte[7];
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, 6, zipEncoding0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(6, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      // Undeclared exception!
      try { 
        TarUtils.formatUnsignedOctalString(0, byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -1
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(8, byteArray0, 8, 8);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 14
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[168];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-23), byteArray0, (-23), (-23));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -23 is too large for -23 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(281474976710656L, byteArray0, 0, 0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = new byte[168];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8589934591L, byteArray0, 121, 32);
      assertEquals(153, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = new byte[168];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-665L), byteArray0, 118, 32);
      assertEquals(150, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = new byte[168];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(134, byteArray0, 134, 32);
      assertEquals(166, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)91;
      byteArray0[1] = (byte) (-35);
      byteArray0[2] = (byte) (-56);
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}