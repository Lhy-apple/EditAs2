/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:33:20 GMT 2023
 */

package org.apache.commons.compress.changes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PipedInputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ar.ArArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream;
import org.apache.commons.compress.changes.ChangeSet;
import org.apache.commons.compress.changes.ChangeSetPerformer;
import org.apache.commons.compress.changes.ChangeSetResults;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ChangeSetPerformer_ESTest extends ChangeSetPerformer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      ArArchiveEntry arArchiveEntry0 = new ArArchiveEntry("p?Mx=", 0L, 1, 1, 1, 0L);
      PipedInputStream pipedInputStream0 = new PipedInputStream(180);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      changeSet0.add((ArchiveEntry) arArchiveEntry0, (InputStream) tarArchiveInputStream0, true);
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 4, 1);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      // Undeclared exception!
      try { 
        changeSetPerformer0.perform(tarArchiveInputStream0, tarArchiveOutputStream0);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.apache.commons.compress.archivers.ar.ArArchiveEntry cannot be cast to org.apache.commons.compress.archivers.tar.TarArchiveEntry
         //
         verifyException("org.apache.commons.compress.archivers.tar.TarArchiveOutputStream", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      changeSet0.delete("H5A*");
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream((InputStream) null);
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 2512, 2512);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      JarArchiveInputStream jarArchiveInputStream0 = new JarArchiveInputStream(tarArchiveInputStream0);
      ChangeSetResults changeSetResults0 = changeSetPerformer0.perform(jarArchiveInputStream0, tarArchiveOutputStream0);
      assertNotNull(changeSetResults0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      ChangeSet changeSet0 = new ChangeSet();
      ArArchiveEntry arArchiveEntry0 = new ArArchiveEntry((String) null, 0L, 2, 0, 2, 1);
      PipedInputStream pipedInputStream0 = new PipedInputStream(1);
      TarArchiveInputStream tarArchiveInputStream0 = new TarArchiveInputStream(pipedInputStream0);
      changeSet0.add((ArchiveEntry) arArchiveEntry0, (InputStream) pipedInputStream0, false);
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      TarArchiveOutputStream tarArchiveOutputStream0 = new TarArchiveOutputStream(byteArrayOutputStream0, 1, 2);
      ChangeSetPerformer changeSetPerformer0 = new ChangeSetPerformer(changeSet0);
      try { 
        changeSetPerformer0.perform(tarArchiveInputStream0, tarArchiveOutputStream0);
        fail("Expecting exception: IOException");
      
      } catch(IOException e) {
         //
         // Pipe not connected
         //
         verifyException("java.io.PipedInputStream", e);
      }
  }
}