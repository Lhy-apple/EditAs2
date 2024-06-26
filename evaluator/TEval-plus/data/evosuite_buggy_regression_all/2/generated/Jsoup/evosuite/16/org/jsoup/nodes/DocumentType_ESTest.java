/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:07:17 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.DocumentType;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DocumentType_ESTest extends DocumentType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("", "", "", "");
      String string0 = documentType0.toString();
      assertEquals("#doctype", documentType0.nodeName());
      assertEquals("<!DOCTYPE html>", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("V-v.^R_7", "UXB;to", "Im", "Zacute");
      String string0 = documentType0.toString();
      assertEquals("<!DOCTYPE html PUBLIC \"UXB;to\" Im\">", string0);
      assertEquals("#doctype", documentType0.nodeName());
  }
}
