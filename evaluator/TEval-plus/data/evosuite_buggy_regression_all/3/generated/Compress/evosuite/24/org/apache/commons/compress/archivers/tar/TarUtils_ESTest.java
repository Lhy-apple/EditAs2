/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:28:55 GMT 2023
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
      byte[] byteArray0 = new byte[209];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(156, byteArray0, (byte)0, 156);
      assertEquals(156, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)106, 105);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes((-2586L), byteArray0, (-3337), (-3337));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -2586=1777777777777777772746 will not fit in octal number buffer of length -3339
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[69];
      TarUtils.formatLongOctalOrBinaryBytes((-1L), byteArray0, 9, 9);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 9, 9);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -1 at offset 8 in '\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD' len=9
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((byte)0, byteArray0, (byte)0, (byte)0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // -3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      String string0 = TarUtils.parseName(byteArray0, (int) (byte)0, (int) (byte) (-37));
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, 6713, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 3
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("Cp850", byteArray0, (-19), (-19), zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[30];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(7, byteArray0, 7, 7);
      assertEquals(14, int0);
      
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, 7, 14, zipEncoding0);
      assertEquals("000007 ", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[46];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(5, byteArray0, 5, 5);
      assertEquals(10, int0);
      
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      String string0 = TarUtils.parseName(byteArray0, 0, 10, zipEncoding0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
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
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[30];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 21, 14);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)32;
      byteArray0[1] = (byte)32;
      long long0 = TarUtils.parseOctal(byteArray0, (byte)0, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[32];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(7, byteArray0, 7, 7);
      assertEquals(14, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 7, 14);
      assertEquals(7L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[18];
      byteArray0[0] = (byte)14;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, (byte)14);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 14 at offset 0 in '\u000E{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}{NUL}' len=14
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)63;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 63 at offset 0 in '?{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[64];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-5L), byteArray0, 5, 5);
      assertEquals(10, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 9, 9);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[37];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2128L), byteArray0, 5, 5);
      assertEquals(10, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 5, 10);
      assertEquals((-2339760743907328L), long0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[37];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-2128L), byteArray0, 5, 5);
      assertEquals(10, int0);
      
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 5, 5);
      assertEquals((-2128L), long0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte) (-6);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 0, (byte) (-6));
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[46];
      TarUtils.formatLongOctalOrBinaryBytes((-2105L), byteArray0, 5, 5);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 8, 10);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 8, 10 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[30];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8, byteArray0, 8, 8);
      assertEquals(16, int0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[30];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8589934609L, byteArray0, 14, 7);
      assertEquals(21, int0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[36];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-39), byteArray0, (-39), (-39));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -39 is too large for -39 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[227];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes((-140L), byteArray0, (byte)0, 156);
      assertEquals(156, int0);
      
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte) (-1);
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.verifyCheckSum(byteArray0);
      assertTrue(boolean0);
  }
}