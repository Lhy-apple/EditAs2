/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:46:30 GMT 2023
 */

package org.apache.commons.compress.utils;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.utils.ArchiveUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ArchiveUtils_ESTest extends ArchiveUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      // Undeclared exception!
      try { 
        ArchiveUtils.toAsciiString(byteArray0, (int) (byte)71, (int) (byte)71);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      boolean boolean0 = ArchiveUtils.isEqual((byte[]) null, (-132), (-132), (byte[]) null, (-132), (-132));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      boolean boolean0 = ArchiveUtils.matchAsciiBuffer("", byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      // Undeclared exception!
      try { 
        ArchiveUtils.isEqualWithNull((byte[]) null, 1940, 1940, (byte[]) null, 1940, (-791));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.utils.ArchiveUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, byteArray0, false);
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
      byte[] byteArray0 = ArchiveUtils.toAsciiBytes("");
      assertArrayEquals(new byte[] {}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SevenZArchiveEntry sevenZArchiveEntry0 = new SevenZArchiveEntry();
      String string0 = ArchiveUtils.toString((ArchiveEntry) sevenZArchiveEntry0);
      assertEquals("-       0 null", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0, "");
      String string0 = ArchiveUtils.toString((ArchiveEntry) tarArchiveEntry0);
      assertEquals("d       0 /", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      boolean boolean0 = ArchiveUtils.matchAsciiBuffer("`/i", byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, (int) (byte) (-6), (int) (byte) (-6), byteArray0, 8, (int) (byte) (-3), true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, (int) (byte)0, 1, byteArray0, (-9), (int) (byte)0, true);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[12];
      byteArray0[2] = (byte) (-13);
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, (int) (byte)0, 4, byteArray0, (int) (byte)0, (int) (byte)0, true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[2] = (byte) (-6);
      boolean boolean0 = ArchiveUtils.isEqual(byteArray0, (int) (byte) (-6), (int) (byte) (-6), byteArray0, 8, (int) (byte) (-3), true);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      boolean boolean0 = ArchiveUtils.isArrayZero(byteArray0, (byte)2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[2] = (byte) (-111);
      boolean boolean0 = ArchiveUtils.isArrayZero(byteArray0, 3298);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = ArchiveUtils.sanitize("1JIeO|HGyQf8");
      assertEquals("1JIeO|HGyQ?f8", string0);
  }
}
