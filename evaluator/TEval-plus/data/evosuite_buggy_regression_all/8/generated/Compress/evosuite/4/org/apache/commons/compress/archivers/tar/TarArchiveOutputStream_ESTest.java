/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:40:06 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.IOException;
import java.io.OutputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
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
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      tarArchiveOutputStream0.flush();
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream((OutputStream) null, 4456);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", "org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      ArchiveEntry archiveEntry0 = tarArchiveOutputStream0.createArchiveEntry(mockFile0, "org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      tarArchiveOutputStream0.putArchiveEntry(archiveEntry0);
      try { 
        tarArchiveOutputStream0.close();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("pwTk$");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", "org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name 'data/swf/zenodo_replication_package_new/org.apache.commons.compress.archivers.zip.ZipArchiveEntry/org.apache.commons.compress.archivers.zip.ZipArchiveEntry' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", "org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      tarArchiveOutputStream0.setLongFileMode(2);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(1000, TarArchiveEntry.MILLIS_PER_SECOND);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("org.apache.commons.compress.archivers.zip.ZipArchiveEntry", "org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      tarArchiveOutputStream0.setLongFileMode(1);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertFalse(tarArchiveEntry0.isGNULongNameEntry());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry(mockFile0);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals("", tarArchiveEntry0.getLinkName());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("pwTk$");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockFileOutputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("org.apache.commons.compress.archivers.zip.ZipArchiveEntry");
      tarArchiveEntry0.setSize(3892L);
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry 'org.apache.commons.compress.archivers.zip.ZipArchiveEntry' closed at '0' before the '3892' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("pwTk$");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      byte[] byteArray0 = new byte[5];
      try { 
        tarArchiveOutputStream0.write(byteArray0, (int) (byte) (-1), 392);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '392' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }
}
