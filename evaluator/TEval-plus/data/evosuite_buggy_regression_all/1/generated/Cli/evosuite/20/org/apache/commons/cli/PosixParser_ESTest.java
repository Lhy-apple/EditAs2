/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:01:10 GMT 2023
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
      stringArray0[0] = "-A";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(2, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = Locale.getISOCountries();
      String[] stringArray1 = posixParser0.flatten((Options) null, stringArray0, true);
      String[] stringArray2 = posixParser0.flatten((Options) null, stringArray1, true);
      assertEquals(252, stringArray2.length);
      assertEquals(251, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "--D\"z=md";
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
  public void test03()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-";
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
  public void test04()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-D\"z=m'd";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(4, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = Locale.getISOCountries();
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, false);
      assertNotSame(stringArray1, stringArray0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("A", "-A", true, "A");
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-A";
      stringArray0[1] = "A";
      posixParser0.flatten(options1, stringArray0, true);
      posixParser0.burstToken("-A", false);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("A", false, "A");
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[10];
      stringArray0[0] = "-A";
      stringArray0[1] = "A";
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, true);
      assertEquals(11, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-W";
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
  public void test09()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("A", "A", false, "A");
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-AD";
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, false);
      assertEquals(2, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("A", "A", true, "A");
      PosixParser posixParser0 = new PosixParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-AD";
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, true);
      assertEquals(2, stringArray1.length);
  }
}
