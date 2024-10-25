/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:27:00 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.PipedOutputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      tarArchiveOutputStream0.flush();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0, 1000);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockFile mockFile0 = new MockFile("X7jV)/WyV=XE<KzF", "X7jV)/WyV=XE<KzF");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 2090, 2090);
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "");
      assertEquals(31, TarArchiveEntry.MAX_NAMELEN);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("4r-E?()q", true);
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFileOutputStream0);
      DataOutputStream dataOutputStream0 = new DataOutputStream(mockPrintStream0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(dataOutputStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.ar.ArArchiveEntry", "org.apache.commons.compress.archivers.ar.ArArchiveEntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(2);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("", tarArchiveEntry0.getLinkName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.ar.ArArchiveEntry", "org.apache.commons.compress.archivers.ar.ArArchiveEntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/lhy/TEval-plus/org.apache.commons.compress.archivers.ar.ArArchiveEntry/org.apache.commons.compress.archivers.ar.ArArchiveEntry' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.ar.ArArchiveEntry", "org.apache.commons.compress.archivers.ar.ArArchiveEntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(1);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(33188, TarArchiveEntry.DEFAULT_FILE_MODE);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("/", tarArchiveEntry0.getName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("X7jV)/WyV=XE<KzF", "X7jV)/WyV=XE<KzF");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 2090, 2090);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFile mockFile0 = new MockFile("X7jV)/WyV=XE<KzF", "X7jV)/WyV=XE<KzF");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 2090, 2090);
      mockPrintStream0.println();
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'data/lhy/TEval-plus/X7jV)/WyV=XE<KzF/X7jV)/WyV=XE<KzF' closed at '0' before the '1' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PipedOutputStream pipedOutputStream0 = new PipedOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(pipedOutputStream0);
      try { 
        tarArchiveOutputStream0.write((byte[]) null, 2, 2);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '2' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }
}
