/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:39:33 GMT 2023
 */

package org.apache.commons.compress.archivers.cpio;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.cpio.CpioArchiveEntry;
import org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveOutputStream_ESTest extends CpioArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      File file0 = MockFile.createTempFile(" /W4F]zS)1gx", " /W4F]zS)1gx");
      ArchiveEntry archiveEntry0 = cpioArchiveOutputStream0.createArchiveEntry(file0, " /W4F]zS)1gx");
      assertEquals(" /W4F]zS)1gx", archiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)8);
      cpioArchiveOutputStream0.finish();
      assertEquals(38, byteArrayOutputStream0.size());
      assertEquals("q\uFFFD\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0001\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u000B\u0000\u0000\u0000\u0000TRAILER!!!\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)4);
      cpioArchiveOutputStream0.finish();
      assertEquals("0707070000000000000000000000000000000000010000000000000000000001300000000000TRAILER!!!\u0000", byteArrayOutputStream0.toString());
      assertEquals(87, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("", 544L);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // invalid entry size (expected 544 but got 0 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      cpioArchiveOutputStream0.close();
      cpioArchiveOutputStream0.close();
      assertEquals(124, byteArrayOutputStream0.size());
      assertEquals("07070200000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)3);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 3
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)5);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 5
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)6);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 6
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)7);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 7
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)17);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 17
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      cpioArchiveOutputStream0.close();
      try { 
        cpioArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Stream closed
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("UTF-8");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // duplicate entry: UTF-8
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(pipedOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry((short)2);
      try { 
        cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Header format: 2 does not match existing format: 1
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
        cpioArchiveOutputStream0.write((byte[]) null, (-1531), (-3456));
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
      CpioArchiveOutputStream cpioArchiveOutputStream1 = new CpioArchiveOutputStream(cpioArchiveOutputStream0);
      try { 
        cpioArchiveOutputStream1.finish();
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
      MockPrintStream mockPrintStream0 = new MockPrintStream("Streamclosed");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(mockPrintStream0);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write(byteArray0, (int) (short)8, (-444));
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
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write(byteArray0, 255, 255);
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
      byteArrayOutputStream0.writeTo(cpioArchiveOutputStream0);
      assertEquals(0, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("", 544L);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      CpioArchiveOutputStream cpioArchiveOutputStream1 = new CpioArchiveOutputStream(cpioArchiveOutputStream0);
      cpioArchiveOutputStream1.finish();
      assertEquals(236, byteArrayOutputStream0.size());
      assertEquals("07070100000000000000000000000000000000000000000000000000000220000000000000000000000000000000000000000100000000\u0000\u000007070100000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("");
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.write(1706);
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
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(" bytes)");
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry(" bytes)");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(mockFileOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream(" bytes)");
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry(" bytes)");
      cpioArchiveEntry0.setTime((-46L));
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(mockFileOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      assertEquals((short)1, cpioArchiveEntry0.getFormat());
  }
}
