/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:34:03 GMT 2023
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
      posixParser0.parse(options0, (String[]) null);
      options0.addOption("1", "1", true, "1");
      posixParser0.burstToken("-[ Options: [ short java.util.HashMap@0000000004 ] [ long {1=[ option: null 1 +ARG :: null ]} ]", true);
      posixParser0.burstToken("-[ Options: [ short java.util.HashMap@0000000004 ] [ long {1=[ option: null 1 +ARG :: null ]} ]", true);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      String[] stringArray1 = posixParser0.flatten((Options) null, stringArray0, false);
      assertNotSame(stringArray0, stringArray1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--org.apache.commons.cli.PosixParser";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "--`~=rf=";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option("o", "org.apache.commons.cli.PosixParser", true, "[ Options: [ short java.util.HashMap@0000000002 ] [ long {} ]");
      options0.addOption(option0);
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[7];
      stringArray0[0] = "-org.apache.commons.cli.PosixParser";
      stringArray0[1] = "[ Options: [ short java.util.HashMap@0000000002 ] [ long {} ]";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(8, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "0";
      stringArray0[1] = "-0";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      posixParser0.parse(options0, (String[]) null, true);
      options0.addOption("1", false, "1");
      posixParser0.burstToken("-[ Optons: [ short java.util.ashMab@0000000004 ]o[ =ong {1=[ option: null 1 +ARG :: null ]} ]", true);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("0", "-0", false, "-0");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "0";
      stringArray0[1] = "-0";
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, false);
      assertEquals(2, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "-1";
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(5, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "1";
      options0.addOption("1", true, "1");
      posixParser0.flatten(options0, stringArray0, true);
      posixParser0.burstToken("-1", true);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "->iZ'1dj";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertEquals(7, stringArray1.length);
  }
}
