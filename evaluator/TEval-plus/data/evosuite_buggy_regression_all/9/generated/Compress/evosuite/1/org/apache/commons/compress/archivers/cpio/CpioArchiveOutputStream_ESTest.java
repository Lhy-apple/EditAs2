/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:19:46 GMT 2023
 */

package org.apache.commons.compress.archivers.cpio;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.cpio.CpioArchiveEntry;
import org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveOutputStream_ESTest extends CpioArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.putArchiveEntry((ArchiveEntry) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)8);
      cpioArchiveOutputStream0.finish();
      assertEquals(38, byteArrayOutputStream0.size());
      assertEquals("q\uFFFD\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0001\u0000\u0000\uFFFD\uFFFD\uFFFD\uFFFD\u0000\u000B\u0000\u0000\u0000\u0000TRAILER!!!\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveOutputStream cpioArchiveOutputStream1 = new CpioArchiveOutputStream(cpioArchiveOutputStream0);
      cpioArchiveOutputStream1.finish();
      assertEquals(124, byteArrayOutputStream0.size());
      assertEquals("0707010000000000000000000000000000000000000001ffffffff00000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)4);
      cpioArchiveOutputStream0.finish();
      assertEquals("0707070000000000000000000000000000000000010000007777777777700001300000000000TRAILER!!!\u0000", byteArrayOutputStream0.toString());
      assertEquals(87, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null);
      cpioArchiveOutputStream0.close();
      try { 
        cpioArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      cpioArchiveOutputStream0.finish();
      assertEquals("0707020000000000000000000000000000000000000001ffffffff00000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
      assertEquals(124, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)3);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown header type
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown header type
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)6);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown header type
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)7);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown header type
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null, (short)1307);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown header type
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("KP8`GJzMKL@tV");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(mockFileOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("KP8`GJzMKL@tV");
      cpioArchiveOutputStream0.putNextEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.putNextEntry(cpioArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // duplicate entry: KP8`GJzMKL@tV
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("ZYm?wV^L6}k-hr%[", (short)8);
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)8);
      cpioArchiveOutputStream0.putNextEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // invalid entry size (expected 8 but got 0 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write((byte[]) null, (-1135), (-1135));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      byte[] byteArray0 = new byte[1];
      try { 
        cpioArchiveOutputStream0.write(byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // no current CPIO entry
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream(0);
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write((byte[]) null, 0, (-3607));
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)8);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write(byteArray0, (int) (short)8, (int) (short)8);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      byte[] byteArray0 = new byte[0];
      cpioArchiveOutputStream0.write(byteArray0);
      assertArrayEquals(new byte[] {}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("org.apache.commons.compress.archivers.cpio.CpioUtil", 442L);
      cpioArchiveOutputStream0.putNextEntry(cpioArchiveEntry0);
      byte[] byteArray0 = new byte[1];
      cpioArchiveOutputStream0.write(byteArray0);
      assertEquals("07070100000000ffffffff000000000000000000000000320f8328000001ba000000000000000000000000000000000000003400000000org.apache.commons.compress.archivers.cpio.CpioUtil\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
      assertEquals(165, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("|-l2/H'i>D@>H");
      cpioArchiveOutputStream0.putNextEntry(cpioArchiveEntry0);
      byte[] byteArray0 = new byte[4];
      try { 
        cpioArchiveOutputStream0.write(byteArray0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // attempt to write past end of STORED entry
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null);
      cpioArchiveOutputStream0.close();
      cpioArchiveOutputStream0.close();
  }
}