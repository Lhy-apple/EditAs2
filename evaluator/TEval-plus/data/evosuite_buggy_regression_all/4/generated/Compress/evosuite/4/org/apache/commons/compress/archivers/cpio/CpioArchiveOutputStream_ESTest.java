/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:28:26 GMT 2023
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
import org.evosuite.runtime.mock.java.io.MockFile;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CpioArchiveOutputStream_ESTest extends CpioArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("");
      ArchiveEntry archiveEntry0 = cpioArchiveOutputStream0.createArchiveEntry(mockFile0, "");
      assertEquals("", archiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)8);
      cpioArchiveOutputStream0.close();
      cpioArchiveOutputStream0.close();
      assertEquals("q\uFFFD\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0001\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u000B\u0000\u0000\u0000\u0000TRAILER!!!\u0000\u0000", byteArrayOutputStream0.toString());
      assertEquals(38, byteArrayOutputStream0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)4);
      cpioArchiveOutputStream0.close();
      assertEquals(87, byteArrayOutputStream0.size());
      assertEquals("0707070000000000000000000000000000000000010000000000000000000001300000000000TRAILER!!!\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      cpioArchiveOutputStream0.finish();
      assertEquals(124, byteArrayOutputStream0.size());
      assertEquals("07070200000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null, (short)3);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 3
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
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
  public void test06()  throws Throwable  {
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
  public void test07()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null, (short)7);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: 7
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CpioArchiveOutputStream cpioArchiveOutputStream0 = null;
      try {
        cpioArchiveOutputStream0 = new CpioArchiveOutputStream((OutputStream) null, (short) (-2896));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Unknown format: -2896
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
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
  public void test10()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // duplicate entry: 
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
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
  public void test12()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("-:6I", "-:6I");
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry(mockFile0, "-:6I");
      cpioArchiveEntry0.setSize(1292L);
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // invalid entry size (expected 1292 but got 0 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write(byteArray0, (-4385), (-4385));
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
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write((byte[]) null, 2487, (-1801));
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
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      byte[] byteArray0 = new byte[0];
      // Undeclared exception!
      try { 
        cpioArchiveOutputStream0.write(byteArray0, (int) (short)2, (int) (short)2);
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
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0, (short)2);
      byte[] byteArray0 = new byte[0];
      cpioArchiveOutputStream0.write(byteArray0);
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("HTp)'n_?`");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveOutputStream cpioArchiveOutputStream1 = new CpioArchiveOutputStream(cpioArchiveOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      try { 
        cpioArchiveOutputStream1.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // attempt to write past end of STORED entry
         //
         verifyException("org.apache.commons.compress.archivers.cpio.CpioArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("HTp)'n_?`");
      cpioArchiveEntry0.setSize(1292L);
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      CpioArchiveOutputStream cpioArchiveOutputStream1 = new CpioArchiveOutputStream(cpioArchiveOutputStream0);
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      cpioArchiveOutputStream1.finish();
      assertEquals(244, byteArrayOutputStream0.size());
      assertEquals("0707010000000000000000000000000000000000000000000000000000050c000000000000000000000000000000000000000a00000000HTp)'n_?`\u000007070100000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000b00000000TRAILER!!!\u0000\u0000\u0000\u0000", byteArrayOutputStream0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("oJD ");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
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
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      CpioArchiveEntry cpioArchiveEntry0 = new CpioArchiveEntry("8YUF1}z{");
      CpioArchiveOutputStream cpioArchiveOutputStream0 = new CpioArchiveOutputStream(byteArrayOutputStream0);
      cpioArchiveEntry0.setTime((-5097L));
      cpioArchiveOutputStream0.putArchiveEntry(cpioArchiveEntry0);
      assertEquals((-5097L), cpioArchiveEntry0.getTime());
      assertEquals(120, byteArrayOutputStream0.size());
  }
}
