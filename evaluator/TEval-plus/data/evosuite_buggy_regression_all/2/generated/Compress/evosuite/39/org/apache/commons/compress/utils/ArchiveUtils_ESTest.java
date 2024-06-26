/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:48:13 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.dump.DumpArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.utils.ArchiveUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveUtils_ESTest extends ArchiveUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      // Undeclared exception!
      try { 
        ArchiveUtils.toAsciiString(byteArray0, (int) (byte)21, (int) (byte)21);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        ArchiveUtils.isEqual((byte[]) null, 1102, 1102, (byte[]) null, 1102, 1102);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = ArchiveUtils.matchAsciiBuffer("", byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        ArchiveUtils.isEqualWithNull(byteArray0, 568, 568, byteArray0, 568, (-12));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 556
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      String string0 = ArchiveUtils.toAsciiString(byteArray0);
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("C1?");
      assertArrayEquals(new byte[] {(byte)67, (byte)49, (byte)63}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DumpArchiveEntry dumpArchiveEntry0 = new DumpArchiveEntry();
      String string0 = ArchiveUtils.toString((ArchiveEntry) dumpArchiveEntry0);
      assertEquals("-       0 null", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("|cCD|*55![Fc/", (byte)30, false);
      String string0 = ArchiveUtils.toString((ArchiveEntry) tarArchiveEntry0);
      assertEquals("d       0 |cCD|*55![Fc/", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      boolean boolean0 = ArchiveUtils.matchAsciiBuffer("=", byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = new byte[2];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[19];
      byte[] byteArray1 = new byte[2];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[2] = (byte) (-128);
      byte[] byteArray1 = new byte[2];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)7;
      byte[] byteArray1 = new byte[2];
      byteArray1[0] = (byte)7;
      byteArray1[1] = (byte)7;
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray1, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      boolean boolean0 = ArchiveUtils.isArrayZero(byteArray0, (-5334));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[2] = (byte) (-128);
      boolean boolean0 = ArchiveUtils.isArrayZero(byteArray0, (byte)92);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = ArchiveUtils.sanitize("(<MLzgQa,*3Nr@");
      assertEquals("(?<MLzgQa,*3Nr@", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = ArchiveUtils.sanitize("\u0000\uFFFD\u0000\u0000\u0000\u0000\u0000\u0000\u0000");
      assertEquals("?????????", string0);
  }
}
