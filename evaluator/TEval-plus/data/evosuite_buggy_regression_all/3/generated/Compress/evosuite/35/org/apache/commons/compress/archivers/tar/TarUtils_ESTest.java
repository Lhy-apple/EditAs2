/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:29:31 GMT 2023
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
      byte[] byteArray0 = new byte[9];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((byte)2, byteArray0, (byte)2, (byte)2);
      assertEquals(4, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 3, 4);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)50, (byte)32, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(63, (byte[]) null, 63, 63);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[14];
      TarUtils.formatLongOctalOrBinaryBytes((-2743L), byteArray0, (byte)4, (byte)4);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 7, 7);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 73 at offset 0 in 'I{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}' len=7
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes(7L, byteArray0, (-1649), (byte)0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 7=7 will not fit in octal number buffer of length -2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte) (-105), (int) (byte) (-105));
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("IBM850", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[4] = (byte)4;
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)4, (int) (byte)4, zipEncoding0);
      assertEquals("\u0004", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[3] = (byte)4;
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, (int) (byte)4, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[216];
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (-2147483518), (byte) (-41));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Length -41 must be at least 2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[216];
      long long0 = TarUtils.parseOctal(byteArray0, 125, 13);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[15];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((byte)0, byteArray0, (byte)4, (byte)4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, (byte)4, (byte)4);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[4] = (byte)4;
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, (byte)4, (byte)4);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 4 at offset 0 in '\u0004{NUL}{NUL}{NUL}' len=4
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[216];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(1099511627776L, byteArray0, 13, 125);
      assertEquals(138, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 13, 125);
      assertEquals(1099511627776L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[15];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-1346L), byteArray0, (byte)4, (byte)4);
      assertEquals(8, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)4, 8);
      assertEquals((-5781025980416L), long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte) (-2);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte) (-2));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[181];
      TarUtils.formatLongOctalOrBinaryBytes((-2711L), byteArray0, 42, 42);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 42, 84);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 42, 84 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[162];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2711L), byteArray0, 60, 60);
      assertEquals(120, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 60, 60);
      assertEquals((-2711L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("MvdK50", byteArray0, (int) (byte) (-122), (int) (byte) (-122), zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, 1061, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 9
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[158];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(135, byteArray0, 135, 21);
      assertEquals(156, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[168];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8, byteArray0, 8, 8);
      assertEquals(16, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[11];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((byte) (-103), byteArray0, (byte) (-103), (byte) (-103));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -103 is too large for -103 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[100];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(16777216L, byteArray0, 8, 8);
      assertEquals(16, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[159];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2733L), byteArray0, 76, 76);
      assertEquals(152, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)44;
      byteArray0[1] = (byte) (-44);
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}