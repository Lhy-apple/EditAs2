/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:25:42 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GnuParser_ESTest extends GnuParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-M|]N$JdL";
      stringArray0[1] = "-";
      try { 
        gnuParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -M|]N$JdL
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "--";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      Options options1 = options0.addOption("M", "M", false, "MF");
      stringArray0[0] = "-M|$JdL";
      String[] stringArray1 = gnuParser0.flatten(options1, stringArray0, true);
      String[] stringArray2 = gnuParser0.flatten(options0, stringArray1, true);
      assertEquals(2, stringArray2.length);
  }
}