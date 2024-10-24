/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:05:11 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PosixParser_ESTest extends PosixParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "?";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(4, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "---.9$%3DSX6R!=cru)Q?0";
      // Undeclared exception!
      try { 
        posixParser0.flatten((Options) null, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--.#BfN5k$R*";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-";
      PosixParser posixParser0 = new PosixParser();
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      stringArray0[1] = ".#BfN5k$R*";
      stringArray0[2] = "-.#BfN5k$R*";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "";
      stringArray0[3] = "";
      stringArray0[4] = "";
      stringArray0[5] = "-.";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(6, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Option option0 = new Option("", ".#BfN5k$R*", false, ".#BfN5k$R*");
      options0.addOption(option0);
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      stringArray0[1] = ".#BfN5k$R*";
      stringArray0[2] = "-.#BfN5k$R*";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("", "", true, "");
      String[] stringArray0 = new String[0];
      posixParser0.flatten(options0, stringArray0, true);
      posixParser0.burstToken("--yo", true);
      posixParser0.burstToken("s=2X:k+", true);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("", "", false, "");
      String[] stringArray0 = new String[0];
      posixParser0.flatten(options0, stringArray0, false);
      posixParser0.burstToken("$-c]2Wta$^", true);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "";
      stringArray0[3] = "";
      stringArray0[4] = "";
      stringArray0[5] = "-.";
      Option option0 = new Option("", ".", true, "--");
      options0.addOption(option0);
      posixParser0.flatten(options0, stringArray0, false);
      posixParser0.burstToken("-.", false);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-\"";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      posixParser0.burstToken("", true);
  }
}
