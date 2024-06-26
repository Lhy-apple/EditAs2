/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 01:59:31 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Locale;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PosixParser_ESTest extends PosixParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "--[ Options: [ short java.util.HashMap@0000000004 ] [ long {=[ option:    [ARG] ::  ]} ]";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(3, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", true, "");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(5, stringArray1.length);
      
      String[] stringArray2 = posixParser0.flatten(options1, stringArray1, true);
      assertEquals(6, stringArray2.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "-";
      PosixParser posixParser0 = new PosixParser();
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-A";
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(4, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-ZA";
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(2, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-ZA";
      PosixParser posixParser0 = new PosixParser();
      Options options1 = options0.addOption("ZA", true, "zAoOy5Bc`)");
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, true);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = Locale.getISOLanguages();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertFalse(stringArray1.equals((Object)stringArray0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-A";
      PosixParser posixParser0 = new PosixParser();
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("", "", true, "");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      posixParser0.parse(options0, stringArray0, true);
      posixParser0.burstToken("--/'9z[^q5GN]5(Z)8L3", false);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      Options options1 = options0.addOption("", "$H,", false, "$H,");
      posixParser0.parse(options1, stringArray0, true);
      posixParser0.burstToken("--/'9z[^q5GN]5(Z)8L3", false);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("", "", true, "");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      posixParser0.parse(options1, stringArray0, true);
      posixParser0.burstToken("--", false);
  }
}
