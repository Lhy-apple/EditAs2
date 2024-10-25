/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:19:52 GMT 2023
 */

package org.apache.commons.compress.archivers.tar;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFile;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class TarArchiveOutputStream_ESTest extends TarArchiveOutputStream_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.flush();
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      TarArchiveOutputStream tarArchiveOutputStream1 = new TarArchiveOutputStream(tarArchiveOutputStream0, 0);
      assertFalse(tarArchiveOutputStream1.equals((Object)tarArchiveOutputStream0));
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      int int0 = tarArchiveOutputStream0.getRecordSize();
      assertEquals(512, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.setLongFileMode(1);
      assertEquals(1, TarArchiveOutputStream.LONGFILE_TRUNCATE);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      tarArchiveOutputStream0.close();
      try { 
        byteArrayOutputStream0.writeTo(tarArchiveOutputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // request to write '10240' bytes exceeds size in header of '0' bytes for entry 'null'
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      TarArchiveEntry tarArchiveEntry0 = new TarArchiveEntry("");
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      try { 
        tarArchiveOutputStream0.finish();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // This archives contains unclosed entries.
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream(" Masked: ");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0, 512, 512);
      tarArchiveOutputStream0.close();
      tarArchiveOutputStream0.close();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockFile mockFile0 = new MockFile("org.apach.commons.compress.archivers.tar.TarArchiveEntry", "org.apach.commons.compress.archivers.tar.TarArchiveEntry");
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      ArchiveEntry archiveEntry0 = tarArchiveOutputStream0.createArchiveEntry(mockFile0, "/data/swf/zenodo_replication_package_new/org.apach.commons.compress.archivers.tar.TarArchiveEntry/org.apach.commons.compress.archivers.tar.TarArchiveEntry");
      // Undeclared exception!
      try { 
        tarArchiveOutputStream0.putArchiveEntry(archiveEntry0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // file name '/data/swf/zenodo_replication_package_new/org.apach.commons.compress.archivers.tar.TarArchiveEntry/org.apach.commons.compress.archivers.tar.TarArchiveEntry' is too long ( > 100 bytes)
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      MockFile mockFile0 = new MockFile("", "");
      TarArchiveEntry tarArchiveEntry0 = (TarArchiveEntry)tarArchiveOutputStream0.createArchiveEntry(mockFile0, "");
      tarArchiveOutputStream0.putArchiveEntry(tarArchiveEntry0);
      assertEquals(0, tarArchiveEntry0.getGroupId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockFile mockFile0 = new MockFile("''8.w3VbHe52j[PH");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      tarArchiveOutputStream0.closeArchiveEntry();
      assertEquals(0, TarArchiveOutputStream.LONGFILE_ERROR);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockFile mockFile0 = new MockFile("''8.w3VbHe52j[PH");
      MockPrintStream mockPrintStream0 = new MockPrintStream(mockFile0);
      mockPrintStream0.append('0');
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(mockPrintStream0);
      ArchiveEntry archiveEntry0 = tarArchiveOutputStream0.createArchiveEntry(mockFile0, "''8.w3VbHe52j[PH");
      tarArchiveOutputStream0.putArchiveEntry(archiveEntry0);
      try { 
        tarArchiveOutputStream0.closeArchiveEntry();
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // entry '''8.w3VbHe52j[PH' closed at '0' before the '1' bytes specified in the header were written
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0);
      byteArrayOutputStream0.writeTo(tarArchiveOutputStream0);
      assertEquals(2, TarArchiveOutputStream.LONGFILE_GNU);
  }
}
