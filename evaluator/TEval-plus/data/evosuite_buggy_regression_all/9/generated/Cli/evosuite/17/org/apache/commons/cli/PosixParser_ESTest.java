/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:36:07 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PosixParser_ESTest extends PosixParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "-Z";
      stringArray0[1] = "-Z";
      stringArray0[2] = "--WF";
      stringArray0[3] = "--@oAX=";
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-Z";
      stringArray0[1] = "`+12's>x'";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-VG";
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Option option0 = new Option("VG", "7tCRfkOjx-6rmVs");
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-VG";
      Options options0 = new Options();
      options0.addOption(option0);
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("Z", "org.apache.commons.cli.UnrecognizedOptionException", true, "-Z");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-Z";
      stringArray0[1] = "`+12's>x'";
      posixParser0.flatten(options0, stringArray0, false);
      posixParser0.burstToken("`+12's>x'", true);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("", false, "");
      String[] stringArray0 = new String[0];
      posixParser0.flatten(options0, stringArray0, false);
      posixParser0.burstToken("--G", true);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-Z";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(3, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = Locale.getISOCountries();
      Options options1 = options0.addOption("", "", true, "");
      posixParser0.flatten(options1, stringArray0, true);
      posixParser0.burstToken("*+-4{", true);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = Locale.getISOCountries();
      Options options1 = options0.addOption("", "", true, "");
      posixParser0.flatten(options1, stringArray0, true);
      posixParser0.burstToken("i+0:$=Rg)-", true);
  }
}