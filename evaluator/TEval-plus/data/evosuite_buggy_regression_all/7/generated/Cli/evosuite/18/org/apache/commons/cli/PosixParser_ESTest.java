/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 17:45:30 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.Properties;
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
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "--[ Options: [ short java.util.HashMap@0000000004 ] [ long {Qi'Z_XCzJDaxC>9\"=[ option: A Qi'Z_XCzJDaxC>9\"  [ARG] :: -b3 ]} ]";
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
  public void test02()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[31];
      stringArray0[0] = "-b3";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(32, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("b", true, "-b3");
      String[] stringArray0 = new String[31];
      stringArray0[0] = "-b3";
      stringArray0[1] = "b";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options1, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[30];
      stringArray0[0] = "-b";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(29, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("b3", true, "-b3");
      String[] stringArray0 = new String[3];
      stringArray0[0] = "b3";
      stringArray0[1] = "-b3";
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
  public void test07()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Option option0 = new Option("b", "-b3");
      options0.addOption(option0);
      String[] stringArray0 = new String[5];
      stringArray0[0] = "-b3";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(7, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("b", true, "-b");
      String[] stringArray0 = new String[30];
      stringArray0[0] = "-b";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options1, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[9];
      stringArray0[0] = "-I";
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
  public void test10()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "b";
      Properties properties0 = new Properties();
      posixParser0.parse(options0, stringArray0, properties0);
      Option option0 = new Option("b", "b", true, "-b");
      options0.addOption(option0);
      posixParser0.burstToken("-b", true);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-b3";
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
}