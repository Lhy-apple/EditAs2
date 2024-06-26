/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:27:07 GMT 2023
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
      byte[] byteArray0 = new byte[13];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(3, byteArray0, 3, 3);
      assertEquals(6, int0);
      
      long long0 = TarUtils.parseOctal(byteArray0, 3, 6);
      assertEquals(3L, long0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)81);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 6
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatCheckSumOctalBytes(3, (byte[]) null, 3, 3);
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
      byte[] byteArray0 = new byte[10];
      TarUtils.formatLongOctalOrBinaryBytes((-440L), byteArray0, 3, 3);
      // Undeclared exception!
      try { 
        TarUtils.parseOctal(byteArray0, 3, 6);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte -1 at offset 0 in '\uFFFD\uFFFDH{NUL}{NUL}{NUL}' len=6
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatOctalBytes((-1712), byteArray0, (-1712), (-1712));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1712=1777777777777777774520 will not fit in octal number buffer of length -1714
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String string0 = TarUtils.parseName((byte[]) null, (-1), (-1));
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      int int0 = TarUtils.formatNameBytes("", byteArray0, (int) (byte)0, (int) (byte)0, zipEncoding0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ZipEncoding zipEncoding0 = TarUtils.FALLBACK_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("Cp437", (byte[]) null, (-1073741823), 2124, zipEncoding0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
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
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 3, 3);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)32;
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)126;
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid byte 126 at offset 0 in '~{NUL}' len=2
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[29];
      TarUtils.formatLongOctalOrBinaryBytes((-854L), byteArray0, 0, 15);
      // Undeclared exception!
      try { 
        TarUtils.parseOctalOrBinary(byteArray0, 8, 15);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // At offset 8, 15 byte binary number exceeds maximum signed long value
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[17];
      byteArray0[0] = (byte) (-87);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 0, 17);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte) (-1);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
      assertEquals(256L, long0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte) (-117);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, (byte)0, 2);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[24];
      byteArray0[0] = (byte) (-1);
      long long0 = TarUtils.parseOctalOrBinary(byteArray0, 0, 22);
      assertEquals(0L, long0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[1] = (byte)1;
      boolean boolean0 = TarUtils.parseBoolean(byteArray0, (byte)1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[0] = (byte) (-75);
      ZipEncoding zipEncoding0 = TarUtils.DEFAULT_ENCODING;
      // Undeclared exception!
      try { 
        TarUtils.parseName(byteArray0, (int) (byte) (-75), (int) (byte)79, zipEncoding0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      // Undeclared exception!
      try { 
        TarUtils.formatNameBytes("Cp437", (byte[]) null, (-1073741823), (-1073741823));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
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
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
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
  public void test23()  throws Throwable  {
      byte[] byteArray0 = new byte[30];
      int int0 = TarUtils.formatLongOctalOrBinaryBytes(8589934607L, byteArray0, 9, 9);
      assertEquals(18, int0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes((-15), byteArray0, (-15), (-15));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value -15 is too large for -15 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        TarUtils.formatLongOctalOrBinaryBytes(1099511627776L, byteArray0, (-1), (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Value 1099511627776 is too large for -1 byte field.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      long long0 = TarUtils.computeCheckSum(byteArray0);
      assertEquals(0L, long0);
  }
}
